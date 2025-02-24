"""
Martin Großbichler
2025/02/19

class to simulate state space model

\dot{x}(t)=A*x(t)+Bu(t)
y(t)=C*x(t)

where A,B,C are the system matrices
x(t) is the state at the time instant t
y(t) is the output (our observations)
u(t) is the external input

"""
import numpy as np
import matplotlib.pyplot as plt

class StateSpaceSimulator:
    def __init__(self, A, B, C, time_steps, sampling_period):
        self.A = A
        self.B = B
        self.C = C
        self.initial_state = []
        self.time_steps = time_steps
        self.sampling_period = sampling_period
        self.input_sequence = []

    def initalize(self,initial_state,input_sequence):
        self.initial_state = initial_state
        self.input_sequence = input_sequence
        self.time_steps = len(input_sequence)
    
    def simulate(self):
        from numpy.linalg import inv
        I = np.identity(self.A.shape[0])
        Ad = inv(I - self.sampling_period * self.A)
        Bd = Ad * self.sampling_period * self.B
        
        Xd = np.zeros((self.A.shape[0], self.time_steps+1))
        Yd = np.zeros((self.C.shape[0], self.time_steps+1))
        t = np.zeros(self.time_steps+1)
        
        for i in range(self.time_steps):
            if i == 0:
                t[i] = 0
                Xd[:, [i]] = self.initial_state
                Yd[:, [i]] = self.C @ self.initial_state
                x = Ad @ self.initial_state + Bd * self.input_sequence[i]
            else:
                t[i] = t[i - 1] + self.sampling_period
                Xd[:, [i]] = x
                Yd[:, [i]] = self.C @ x
                x = Ad @ x + Bd * self.input_sequence[i]
        Xd[:,[-1]]=x
        Yd[:,[-1]]=self.C*x
        t[-1]=t[-2]+self.sampling_period
        return Xd, Yd, t
    
    def plot_sys_resp(self, time, output):
        plt.plot(time, output[0, :], label='Output')
        plt.plot(time[:-1], self.input_sequence[:,0].T, label='Input', linestyle='dashed')
        plt.xlabel('Discrete time instant-k')
        plt.ylabel('Position - d')
        plt.title('System Step Response')
        plt.legend()
        plt.savefig('step_response.png')
        plt.show()
    
    def prepare_model_training(self):
        Xd,Yd,t = self.simulate()
        output = Yd.T
        output = np.reshape(output, (1, output.shape[0], 1))
        
        input_seq = np.reshape(self.input_sequence, (self.input_sequence.shape[0], 1))
        tmp = np.concatenate((input_seq, np.zeros((input_seq.shape[0], 1))), axis=1)
        tmp = np.concatenate((self.initial_state.T, tmp), axis=0)
        X = np.reshape(tmp, (1, tmp.shape[0], tmp.shape[1]))
        
        return X, output

# Example usage
if __name__ == "__main__":
    A = np.matrix([[0, 1], [-0.1, -0.05]])
    B = np.matrix([[0], [1]])
    C = np.matrix([[1, 0]])
    x0 = np.random.rand(2, 1)
    time_steps = 300
    sampling_period = 0.5
    input_seq = np.ones(time_steps)
    
    simulator = StateSpaceSimulator(A, B, C, x0, time_steps, sampling_period, input_seq)
    state, output, time = simulator.simulate()
    simulator.plot_sys_resp(time,output)
    trainX, output_train = simulator.model_training_prep()
