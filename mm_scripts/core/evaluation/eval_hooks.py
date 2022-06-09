import os

from mmdet.core.evaluation import EvalHook


class SaveResEvalHook(EvalHook):
    def evaluate(self, runner, results):
        jsonfile_prefix = os.path.join(runner.work_dir, f"result_epoch_{runner.epoch}")
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, jsonfile_prefix=jsonfile_prefix, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]
        else:
            return None
