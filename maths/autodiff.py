import numpy as np
import pydrake.symbolic as sym


class CostDerivatives:
    def __init__(self, cost_stage, cost_final, xref, uref, n_x, n_u):
        self.x_sym = np.array([sym.Variable(f"x_{i}") for i in range(n_x)])
        self.u_sym = np.array([sym.Variable(f"u_{i}") for i in range(n_u)])
        
        x = self.x_sym
        u = self.u_sym

        self.xref = xref
        self.uref = uref

        self.cost_stage = cost_stage
        self.cost_final = cost_final

        # Compute the cost expressions
        stage_cost = cost_stage(x, u, xref, uref)
        final_cost = cost_final(x, xref)
        
        # Compute Jacobians of the cost wrt x and u
        self.l_x = sym.Jacobian([stage_cost], x).ravel()
        self.l_u = sym.Jacobian([stage_cost], u).ravel()
        
        # Second derivatives (Hessians)
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        # Final cost derivatives
        self.l_final_x = sym.Jacobian([final_cost], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

    def set_references(self, xref, uref):
        # Recalculate Jacobians if the references change
        x = self.x_sym
        u = self.u_sym
        
        stage_cost = self.cost_stage(x, u, xref, uref)
        final_cost = self.cost_final(x, xref)
        
        self.l_x = sym.Jacobian([stage_cost], x).ravel()
        self.l_u = sym.Jacobian([stage_cost], u).ravel()
        
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        self.l_final_x = sym.Jacobian([final_cost], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

    def stage(self, x, u):
        # Create environment with symbolic values
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        
        # Evaluate derivatives (Jacobian and Hessian)
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        return l_x, l_u, l_xx, l_ux, l_uu

    def final(self, x):
        # Create environment with symbolic values for final cost
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        
        # Evaluate final cost derivatives
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx
    
if __name__ == "__main__":
    import maths.quaternions as qmath

    Q = np.diag(np.concatenate(([1.] * 6, [1.] * 6)))
    R = np.diag([0.1] * 4)


    def cost_stage(x, u, xref, uref):
        m = sym if x.dtype == object else np  # Check type for autodiff
        
        q0 = x[:4]

        G = np.zeros((13, 12), x.dtype)
        G[:4, :3] = qmath.GetAttititudeJacobian(q0)
        G[4:, 3:] = np.eye(9)


        # Define the stage cost
        state_cost = 0.5 * (x - xref) @ G @ Q @ G.T @ (x - xref).T
        action_cost = 0.5 * (u - uref) @ R @ (u - uref).T

        # Return the full cost expression
        return state_cost + action_cost

    def cost_final(x, xref):
        m = sym if x.dtype == object else np  # Check type for autodiff


        q0 = x[:4]

        G = np.zeros((13, 12),x.dtype)
        G[:4, :3] = qmath.GetAttititudeJacobian(q0)
        G[4:, 3:] = np.eye(9)

        # Define the final cost (only state cost)
        state_cost = 0.5 * (x - xref) @ G @ Q @ G.T @ (x - xref).T
        return state_cost
    
    # Testing the code with sample input
    xref = np.zeros(13)
    xref[0] = 1  # Set reference state
    uref = np.zeros(4)  # Set reference input

    # Create derivatives object
    derivs = derivatives(cost_stage, cost_final, xref, uref, 13, 4)

    # Test the derivative computations
    print(derivs.stage(np.zeros(13), np.zeros(4)))