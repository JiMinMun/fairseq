# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, weights=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    # print("target shape: {}".format(target.shape))
    # print("lprobs shape: {}".format(lprobs.shape))
    # print("ignore index: {}".format(ignore_index))
    # if target.shape[0] != lprobs.shape[0]:
    #     target = target[:lprobs.shape[0]]
    # print("target.shape: {}".format(target.shape))
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # print("nll_loss shape: {}".format(nll_loss.shape))
    # if weights is not None:
    #     print("weights: {}".format(weights))
    #     nll_loss.mul(weights)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # print("smooth loss shape: {}".format(smooth_loss.shape))
    if ignore_index is not None:
        # zero_mask = torch.zeros(nll_loss.shape, dtype=torch.bool)
        pad_mask = target.eq(ignore_index)
        # print("pad mask shape: {}".format(pad_mask.shape))
        # zero_mask[:min(nll_loss.shape[0], target.shape[0]), :] = pad_mask
        # pad_mask = zero_mask.to(pad_mask.device)
        # print("zero mask shape: {}".format(zero_mask.shape))
        # if lprobs.shape[0] > target.shape[0]:
        #     pad_mask = torch.nn.ConstantPad2d((0, 0, 0, lprobs.shape[0] - target.shape[0]), 0)(pad_mask).bool()
        #     nll_loss = torch.nn.ConstantPad2d((0, 0, 0, lprobs.shape[0] - target.shape[0]), 0.0)(nll_loss)
        # print("pad mask shape: {}".format(pad_mask.shape))
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    if weights != None:
        return loss * weights, nll_loss * weights
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        use_weights=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # print(sample["net_input"])
        # print(type(sample["net_input"]))
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        if sample.get("artificial_target", None) is not None:
            art_sample = sample.copy()
            art_sample["net_input"] = sample["artificial_net_input"]
            art_sample["target"] = sample["artificial_target"]
            net_output = model(**art_sample["net_input"])
            art_loss, art_nll_loss = self.compute_loss(model, net_output, art_sample, reduce=reduce)
            loss = 0.9 * loss + 0.1 * art_loss
            nll_loss = 0.9 * nll_loss + 0.1 * art_nll_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # artificial_target = sample["artificial_target"]
        # print("target shape: {}".format(target.shape))
        # print("artificial target shape: {}".format(artificial_target.shape))
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
                # if artificial_target is not None:
                #     artificial_target = artificial_target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
                # if artificial_target is not None:
                #     artificial_target = artificial_target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        # , artificial_target.view(-1) if artificial_target is not None else None

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        # print("calling label_smoothed_nll_loss")
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            weights=sample['weights']
        )

        # if artificial_target is not None:
        #     art_loss, art_nll_loss = label_smoothed_nll_loss(
        #         lprobs,
        #         artificial_target,
        #         self.eps,
        #         ignore_index=self.padding_idx,
        #         reduce=reduce,
        #         weights=sample['weights']
        #     )
        #     loss = 0.9 * loss + 0.1 * art_loss
        #     nll_loss = 0.9 * nll_loss + 0.1 * art_nll_loss
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
