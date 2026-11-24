# Scientific scope of the reduced EXP 6.2

This repository implements a focused language–dialect transfer study rather than the complete seven-condition three-task grouping grid.

Newly trained models:

- `one_bit/lid_only`
- `one_bit/dialect_only`
- `one_bit/lid_dialect`
- `sixteen_bit/lid_only`
- `sixteen_bit/dialect_only`
- `sixteen_bit/lid_dialect`

Each best checkpoint is evaluated on both the matched and opposite feature domain, giving 12 test evaluations.

Primary directional affinity measures:

- `A(LID | Dialect) = LID(lid_dialect) - LID(lid_only)`
- `A(Dialect | LID) = Dialect(lid_dialect) - Dialect(dialect_only)`

Macro-F1 affinity is primary; accuracy affinity is secondary.

Once EXP 6.1 finishes, its matched three-task results (`LID + dialect + gender`) are added as an external reference. Because EXP 6.1 uses a three-task development-selection score while `lid_dialect` uses a two-task score, that comparison is practical and informative but is not a perfectly isolated causal estimate of gender.

The experiment does not support claims about a complete three-task affinity matrix, gender pairwise transfer, or the globally best pair among all possible task combinations.
