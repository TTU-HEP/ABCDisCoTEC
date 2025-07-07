"""
abcdiscotec package

This package provides functionalities for building a
background estimation model using pairs of neural networks.

The package provides the following functionalities:
- Building a model
- Applying constraints to the model, such as decorrelation
  and closure constraints. The constraints can be implemented
  as lambda functions or using the MDMM method
- Saving and loading a model
"""

from .abcdiscotec import (
    make_abcdiscotec_model,
    save_checkpoint,
    load_checkpoint,
    get_dataloader,
    make_constraint_manager,
    get_optimizer_and_mdmm_module,
    train_model,
    evaluate_model,
    )
from .constraint_manager import (
    LambdaConstraint,
    MdmmConstraint,
    ClosureConstraintLambda,
    ClosureConstraintMDMM,
    DiscoConstraintLambda,
    DiscoConstraintMDMM,
    )

__version__ = "0.1.0"
__author__ = "The CMS Collaboration"