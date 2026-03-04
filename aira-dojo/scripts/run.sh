python -m dojo.main_run \
  +_exp=run_example \
  interpreter=python \
  task.name=random-acts-of-pizza \
  solver/client@solver.operators.analyze.llm.client=gdm \
  solver/client@solver.operators.debug.llm.client=gdm \
  solver/client@solver.operators.draft.llm.client=gdm \
  solver/client@solver.operators.improve.llm.client=gdm \
  solver.step_limit=3 \
  logger.use_wandb=False