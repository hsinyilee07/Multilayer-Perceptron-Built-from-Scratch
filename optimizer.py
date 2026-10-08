
class Optimizer:
    pass

class Adam(Optimizer):
  def __init__(self, epsilon = 1e-8, beta1 = 0.9, beta2 = 0.99):
      self.t = 1
      self.epsilon = epsilon
      self.beta1 = beta1
      self.beta2 = beta2

  def __call__(self, p_grad, p_m, p_v):
      #1) p_m and p_v are the weighted weight of the historical weight with curret wieght
      new_p_m = self.beta1 * p_m + (1-self.beta1) * p_grad
      new_p_v = self.beta2 * p_v + (1-self.beta2) * p_grad ** 2

      #2) adjust p_m and p_v to prevent cold start
      p_m_hat = new_p_m / (1 - self.beta1 ** self.t) 
      p_v_hat = new_p_v / (1 - self.beta2 ** self.t)

      #Adjust the gradient using formula
      updated_gradient = p_m_hat / (p_v_hat ** 0.5 + self.epsilon)

      return updated_gradient, new_p_m, new_p_v

  def increment_t(self):  
      self.t += 1
    