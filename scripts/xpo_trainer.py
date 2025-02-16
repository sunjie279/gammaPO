
from trl import DPOTrainer
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from transformers import PreTrainedModel
import random
import collections
import numpy as np

class XPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        training_args = kwargs["args"]
        self.beta = training_args.beta
        self.gamma_beta_ratio = training_args.gamma_beta_ratio
        
        self.length_norm = training_args.length_norm
        self.gamma0 = self.gamma_beta_ratio * self.beta
        self.tau = training_args.tau
        self.flip = training_args.flip
        self.reward_gap_queue_size = 2048
        self.opt_size = 256
        self.eps = 1e-10
        self.reward_gap_queue = collections.deque(np.random.randn(8), maxlen=self.reward_gap_queue_size)
        self.gap_mean = 0   # also used for beta-DPO
        self.gap_std = 1    # also used for beta-DPO

    def get_gamma_dq(self,
                     reward_gap: torch.FloatTensor,
        ) -> torch.FloatTensor:
        # optimize the \gamma in a queue, randomly select reward gaps from previous, only sample from recent `reward_gap_queue_size' 
        p_lr = min(0.5 / self.tau, 1)
        sample_num = min(len(self.reward_gap_queue), self.opt_size-reward_gap.numel())
        batch_size = sample_num + reward_gap.numel()
        sampled_gaps = random.sample(self.reward_gap_queue, sample_num)
        reward_gaps = torch.tensor(sampled_gaps + reward_gap.tolist(), dtype=reward_gap.dtype).to(reward_gap.device)
        p = torch.full_like(reward_gaps, fill_value=1/batch_size).to(reward_gap.device)
        for _ in range(20):
            gamma = self.gamma0 * batch_size * p
            gamma_grad = self.gamma0 * batch_size
            KL_grad = self.tau * (1 + torch.log(batch_size * (p + self.eps)))
            p_grad = KL_grad + gamma_grad / (1 + torch.exp(reward_gaps - gamma)) / batch_size
            with torch.no_grad():
                exp_grad = torch.exp(-p_lr * p_grad)
                p = p * exp_grad / torch.sum(p * exp_grad)
        gamma = gamma.detach()[-reward_gap.numel():]
        reward_gap = reward_gap.detach()
        print(f"[opt \gamma in queue] tau={self.tau} | delta={gamma-self.gamma0} | reward_gaps={reward_gaps.view(-1)[-reward_gap.numel():]}")
        self.reward_gap_queue.extend(reward_gap.tolist())
        return gamma - self.gamma0
    
    def xpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        len_chosen: int,
        len_rejected: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        reference_chosen_logps = reference_chosen_logps.to(self.accelerator.device)
        reference_rejected_logps = reference_rejected_logps.to(self.accelerator.device)
        
        logits = policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps
        if self.flip > 0:
            if random.random() <= self.flip:
                pi_logratios = -pi_logratios

        if self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.gamma_beta_ratio)) ** 2
        
        elif self.loss_type == "kto":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        
        elif self.loss_type == "r-dpo":
            gamma = self.gamma_beta_ratio * (len_chosen - len_rejected)
            logits = logits - gamma
            losses = (
                -F.logsigmoid( self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "dpo":
            print(f"[dpo{'-ls' if self.label_smoothing>0 else ''}] reward_gaps={self.beta * logits}")
            losses = (
                -F.logsigmoid( self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            
        elif self.loss_type == "b-dpo":
            reward_gap = self.beta * logits
            # gather rewards from different gpus
            reward_gaps = [torch.zeros_like(reward_gap) for _ in range(dist.get_world_size())]
            dist.all_gather(reward_gaps, reward_gap)
            reward_gaps = torch.cat(reward_gaps, dim=0).view(-1)
            # select data
            weight_sample = torch.exp(-0.5 * (reward_gaps - self.gap_mean) ** 2 / (self.gap_std+self.eps) ** 2)
            sample_num = int(weight_sample.numel() * 0.8)
            sample_index = torch.multinomial(weight_sample, sample_num, replacement=False)
            global_mask = torch.zeros_like(weight_sample)
            global_mask[sample_index] = 1
            global_mask = global_mask.detach()
            # compute loss
            sampled_gap_mean = torch.mean(reward_gaps[sample_index])
            beta_used = self.beta * (1 + self.tau * (sampled_gap_mean - self.gap_mean))
            beta_used = beta_used.detach().clamp(min=0.001)
            losses = (
                -F.logsigmoid( beta_used * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-beta_used * logits) * self.label_smoothing
            )
            # return gap_mean and gap_std for update them
            gap_mean, gap_std = reward_gaps.mean(), reward_gaps.std()
            if dist.get_rank()==0:
                print(f"beta-dpo | beta_used={beta_used.detach()} | gap_mean={gap_mean.detach()} | reward_gaps={reward_gaps.detach()}")
                
        elif self.loss_type == "gm-dpo":
            reward_gap = self.beta * logits
            logits = logits - self.get_gamma_dq(reward_gap) / self.beta
            losses = (
                -F.logsigmoid( self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['dpo', 'ipo', 'kto', 'r-dpo']"
            )
        # compute the loss
        chosen_rewards = (
            self.beta * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)).detach()
        )
        
        if self.loss_type == "b-dpo":
            return losses, chosen_rewards, rejected_rewards, gap_mean, gap_std
        else:
            return losses, chosen_rewards, rejected_rewards, None, None

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)
        losses, chosen_rewards, rejected_rewards, gap_mean, gap_std = self.xpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            len_chosen=batch["chosen_labels"].shape[0],
            len_rejected=batch["rejected_labels"].shape[0]
        )
        if self.loss_type == "b-dpo":
            self.gap_mean = 0.9 * self.gap_mean + 0.1 * gap_mean
            self.gap_std = 0.9 * self.gap_std + 0.1 * gap_std

        prefix = "eval_" if train_eval == "eval" else ""
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = reference_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = reference_chosen_logps.detach().mean().cpu()

        if isinstance(losses, tuple):
            losses, raito_gamma, ref_logratios = losses
            metrics[f"{prefix}RATIO/gamma"] = raito_gamma
            metrics[f"{prefix}RATIO/ref_logratios"] = ref_logratios.detach().mean().cpu()
            metrics[f"{prefix}RATIO/MIN_ref_logratios"] = ref_logratios.detach().min().cpu()
            metrics[f"{prefix}RATIO/MAX_ref_logratios"] = ref_logratios.detach().max().cpu()
            metrics[f"{prefix}RATIO/var_ref_logratios"] = ref_logratios.detach().var().cpu()
        elif isinstance(losses, torch.Tensor):
            if losses.ndim == 0:
                return losses, metrics
        return losses.mean(), metrics

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.length_norm,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a XPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch