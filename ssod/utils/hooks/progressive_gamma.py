from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ProgressiveGammaHook(Hook):
    def __init__(self, log_interval=50):
        if not isinstance(log_interval, int) or log_interval < 1:
            raise ValueError("PG log_interval must be positive")
        self.log_interval = log_interval

    @staticmethod
    def model(runner):
        return runner.model.module if is_module_wrapper(runner.model) else runner.model

    def before_run(self, runner):
        model = self.model(runner)
        pg = getattr(model, "pg", None)
        if pg is None:
            raise RuntimeError("ProgressiveGammaHook requires a PG model")
        if runner.max_iters != pg.total_iters:
            raise RuntimeError("PG total_iters must match runner.max_iters, including on resume")
        if int(pg.last_iter.item()) != runner.iter - 1:
            raise RuntimeError("PG checkpoint/runner iteration mismatch; use --resume-from, not --load-from")

    def before_train_iter(self, runner):
        model = self.model(runner)
        model.pg.set_iteration(runner.iter)
        sup2, multiplier = model.pg.weights()
        unsup2 = model.unsup_weight * multiplier
        runner.log_buffer.output.update(pg_sup2_weight=sup2, pg_unsup2_weight=unsup2)
        if runner.iter == 0 or (runner.iter + 1) % self.log_interval == 0 or runner.iter + 1 == runner.max_iters:
            runner.logger.info("[PG weights] iter=%d/%d mode=%s sup2=%.9f unsup2=%.9f",
                               runner.iter + 1, runner.max_iters, model.pg.mode, sup2, unsup2)
