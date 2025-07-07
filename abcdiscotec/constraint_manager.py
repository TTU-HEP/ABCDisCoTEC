from abcdiscotec._utilities import distance_corr, closure, get_inputs_for_constraints
import mdmm


class LambdaConstraint():
    def __init__(
        self,
        constraint_name:str,
        lambda_weight:float,
    ):
        self.constraint_name = constraint_name
        self.lambda_weight = lambda_weight

    def get_name(self):
        return self.constraint_name
    
    def constraint(
            self,
            dnn_outputs,
            extra_constraint_inputs,
            constraint_obs_names,
            weights,
            labels,
            ):
        raise NotImplementedError("Constraint function not implemented")


class MdmmConstraint():
    def __init__(
        self,
        constraint_name:str,
        type:str,
        target:float,
        scale:float,
        damping:float,
    ):
        self.constraint_name = constraint_name
        self.type = type
        self.target = target
        self.scale = scale
        self.damping = damping

    def get_name(self):
        return self.constraint_name
    
    def constraint(
            self,
            dnn_outputs,
            extra_constraint_inputs,
            constraint_obs_names,
            weights,
            labels,
            ):
        raise NotImplementedError("Constraint function not implemented")


class DiscoConstraintLambda(LambdaConstraint):
    def __init__(
        self,
        obs_name_1:str,
        obs_name_2:str,
        lambda_weight:float,
    ):
        super(DiscoConstraintLambda, self).__init__(
            constraint_name=f"DisCo_{obs_name_1}_{obs_name_2}",
            lambda_weight=lambda_weight,
        )

        self.argument_names = [obs_name_1, obs_name_2, "weights"]
        self.constraint = lambda dnn_outputs, constr_obs, constr_obs_names, weights, labels: self.lambda_weight*distance_corr(
            *get_inputs_for_constraints(self.argument_names, dnn_outputs, constr_obs, constr_obs_names, weights, labels)
        )


class ClosureConstraintLambda(LambdaConstraint):
    def __init__(
        self,
        obs_name_1:str,
        obs_name_2:str,
        n_events_min:int,
        lambda_weight:float,
        symmetrize:bool=False,
    ):
        super(ClosureConstraintLambda, self).__init__(
            constraint_name=f"Closure_{obs_name_1}_{obs_name_2}",
            lambda_weight=lambda_weight,
        )

        self.constraint = lambda dnn_outputs, constr_obs, constr_obs_names, weights, labels: self.lambda_weight*closure(
            *get_inputs_for_constraints(
                [obs_name_1, obs_name_2, "weights", "labels"],
                dnn_outputs,
                constr_obs,
                constr_obs_names,
                weights,
                labels,
                ),
            symmetrize=symmetrize,
            n_events_min=n_events_min,
        )
        self.argument_names = [obs_name_1, obs_name_2, "weights", "labels"]


class DiscoConstraintMDMM(MdmmConstraint):
    def __init__(
        self,
        obs_name_1:str,
        obs_name_2:str,
        type:str,
        target:float,
        scale:float=1.,
        damping:float=0.,
    ):
        super(DiscoConstraintMDMM, self).__init__(
            constraint_name=f"DisCo_{obs_name_1}_{obs_name_2}",
            type=type,
            target=target,
            scale=scale,
            damping=damping,
        )

        self.argument_names = [obs_name_1, obs_name_2, "weights"]

        args = [
            lambda dnn_outputs, constr_obs, constr_obs_names, weights, labels: distance_corr(
                *get_inputs_for_constraints(
                    self.argument_names,
                    dnn_outputs,
                    constr_obs,
                    constr_obs_names,
                    weights,
                    labels,
                    )),
            target,
            scale,
            damping,
        ]

        if self.type == "min":
            self.constraint = mdmm.MinConstraint(*args)
        elif self.type == "max":
            self.constraint = mdmm.MaxConstraint(*args)
        elif self.type == "equal":
            self.constraint = mdmm.EqConstraint(*args)
        else:
            raise ValueError("Constraint type must be either 'min', 'max' or 'equal'")
        

class ClosureConstraintMDMM(MdmmConstraint):
    def __init__(
        self,
        obs_name_1:str,
        obs_name_2:str,
        n_events_min,
        type:str,
        target:float,
        symmetrize:bool=False,
        scale:float=1.,
        damping:float=0.,
    ):
        super(ClosureConstraintMDMM, self).__init__(
            constraint_name=f"Closure_{obs_name_1}_{obs_name_2}",
            type=type,
            target=target,
            scale=scale,
            damping=damping,
        )
        
        self.argument_names = [obs_name_1, obs_name_2, "weights", "labels"]

        args = [
            lambda dnn_outputs, constr_obs, constr_obs_names, weights, labels: closure(
                *get_inputs_for_constraints(self.argument_names, dnn_outputs, constr_obs, constr_obs_names, weights, labels),
                symmetrize=symmetrize,
                n_events_min=n_events_min,
                ),
            target,
            scale,
            damping,
        ]

        if self.type == "min":
            self.constraint = mdmm.MinConstraint(*args)
        elif self.type == "max":
            self.constraint = mdmm.MaxConstraint(*args)
        elif self.type == "equal":
            self.constraint = mdmm.EqConstraint(*args)
        else:
            raise ValueError("Constraint type must be either 'min', 'max' or 'equal'")


class ConstraintManager():
    """
    Class to manage constraints.

    Args:
        extra_variables_for_constraints (list[str]): The names of the extra variables needed for the constraints
                                                     (any variable that is not one of the DNNs).
    """
    def __init__(self, extra_variables_for_constraints:list[str]):
        self.constraint_obs = extra_variables_for_constraints
        self.lambda_constraint_names = []
        self.lambda_constraints = []
        self.arguments_lambda_constraints = []
        self.mdmm_constraint_names = []
        self.mdmm_constraints = []
        self.arguments_mdmm_constraints = []

    def add_constraint(self, constraint):
        if isinstance(constraint, LambdaConstraint):
            self.lambda_constraints.append(constraint.constraint)
            self.lambda_constraint_names.append(constraint.get_name())
            self.arguments_lambda_constraints.append(constraint.argument_names)
        elif isinstance(constraint, MdmmConstraint):
            self.mdmm_constraints.append(constraint.constraint)
            self.mdmm_constraint_names.append(constraint.get_name())
            self.arguments_mdmm_constraints.append(constraint.argument_names)
        else:
            raise ValueError("Constraint must be either a LambdaConstraint or a MdmmConstraint")

    def get_argument_names(self):
        return self.arguments_lambda_constraints + self.arguments_mdmm_constraints

    def get_lambda_constraints(self):
        return self.lambda_constraints
    
    def get_lambda_constraint_names(self):
        return self.lambda_constraint_names
    
    def get_mdmm_constraints(self):
        return self.mdmm_constraints
    
    def get_mdmm_constraint_names(self):
        return self.mdmm_constraint_names
    
    def make_mdmm_optimizer_and_module(self, model, learning_rate) -> tuple:
        # We build the MDMM module with only the MDMM constraints
        # making sure to pass the lambda constraints together with the bce loss
        mdmm_module = mdmm.MDMM(self.get_mdmm_constraints())
        optimizer = mdmm_module.make_optimizer(model.parameters(), lr=learning_rate)

        return optimizer, mdmm_module
