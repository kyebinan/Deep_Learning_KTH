from util import *
from scipy.special import expit as sigmoid

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 5000

        self.errors = []
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return
    
    def viz_rf(self, weights, it, grid):
        """
        Visualize receptive fields (weights) of the RBM.

        Args:
        weights: A 3D numpy array of weights to visualize. The dimensions should match [image_height, image_width, n_hidden_units].
        it: Iteration number, used for the plot title.
        grid: Tuple indicating the grid size for plotting (rows, columns).
        """
        fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=(grid[1] * 3, grid[0] * 3))
        
        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < weights.shape[2]:  # Check to avoid index error if there are fewer weights than grid slots
                ax.imshow(weights[:, :, i], cmap='gray', interpolation='nearest')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide unused subplots

        plt.suptitle(f'Receptive Fields at Iteration {it}')
        plt.tight_layout()
        plt.show()

        
    def cd1(self, visible_trainset, n_iterations=10000):
        """
        Perform Contrastive Divergence with k=1 full alternating Gibbs sampling.

        Args:
        visible_trainset: Training data for this RBM, shape (size of training set, size of visible layer).
        n_iterations: Number of iterations of learning (each iteration learns a mini-batch).
        """
        # Initialize a list to store reconstruction errors for plotting
        self.errors = []

        print("Starting CD1 training")

        for it in range(n_iterations):
            # Selecting a minibatch randomly
            indices = np.random.choice(range(visible_trainset.shape[0]), size=self.batch_size, replace=False)
            v0 = visible_trainset[indices]

            # Start of Gibbs Sampling
            # Step 1: From visible to hidden
            ph0, h0 = self.get_h_given_v(v0)  # Compute hidden probabilities and sample hidden states
            # Step 2: From hidden to visible (reconstruction)
            pvk, vk = self.get_v_given_h(h0)  # Compute visible probabilities and sample visible states
            # Step 3: From visible back to hidden
            phk, hk = self.get_h_given_v(vk)  # Compute new hidden probabilities and sample hidden states

            # Update parameters
            self.update_params(v0, h0, vk, hk)

            # Calculate and store the reconstruction error
            error = np.linalg.norm(v0 - vk)  # Simple reconstruction error; not the actual cost function
            self.errors.append(error)

            # Visualizing learning progress
            # if it % self.rf["period"] == 0 and self.is_bottom:
            #     self.viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)), it=it, grid=self.rf["grid"])

            # Print progress
            if it % self.print_period == 0:
                print("Iteration: {}, Reconstruction error: {:.4f}".format(it, error))


    

    def update_params(self, v0, h0, vk, hk):
        """
        Update the weight and bias parameters.

        Args:
        v0: Initial visible units states
        h0: Initial hidden units states
        vk: Reconstructed visible units states
        hk: Reconstructed hidden units states
        """
        # Compute positive and negative associations
        positive_assoc = np.dot(v0.T, h0)
        negative_assoc = np.dot(vk.T, hk)

        # Update weights and biases
        self.weight_vh += self.learning_rate * ((positive_assoc - negative_assoc) / self.batch_size)
        self.bias_v += self.learning_rate * np.mean(v0 - vk, axis=0)
        self.bias_h += self.learning_rate * np.mean(h0 - hk, axis=0)
            
        return

    def get_h_given_v(self, visible_minibatch):
        """
        Compute probabilities p(h|v) and activations h ~ p(h|v) for the hidden units given the visible units.

        Args:
        visible_minibatch: A minibatch of the visible units' states, shape (size of mini-batch, size of visible layer).

        Returns:
        A tuple (p_h_given_v, h_samples):
            - p_h_given_v: The probabilities of the hidden units being active given the visible units, shape (size of mini-batch, size of hidden layer).
            - h_samples: Sampled states of the hidden units from the computed probabilities, shape (size of mini-batch, size of hidden layer).
        """
        # Compute the total input to the hidden units.
        total_input = np.dot(visible_minibatch, self.weight_vh) + self.bias_h

        # Compute the probabilities of the hidden units given the visible units.
        p_h_given_v = sigmoid(total_input)

        # Sample the states of the hidden units based on the probabilities.
        # This uses stochastic binary sampling; a hidden unit is turned on (1) if a randomly drawn number is less than the unit's computed probability.
        h_samples = np.random.binomial(n=1, p=p_h_given_v)

        return p_h_given_v, h_samples


    def get_v_given_h(self, hidden_minibatch):
        """
        Compute probabilities p(v|h) and activations v ~ p(v|h) for the visible units given the hidden units.

        Args:
        hidden_minibatch: A minibatch of the hidden units' states, shape (size of mini-batch, size of hidden layer).

        Returns:
        A tuple (p_v_given_h, v_samples):
            - p_v_given_h: The probabilities of the visible units being active given the hidden units, shape (size of mini-batch, size of visible layer).
            - v_samples: Sampled states of the visible units from the computed probabilities, shape (size of mini-batch, size of visible layer).
        """
        
        assert self.weight_vh is not None
        
        # Compute the total input to the visible units.
        total_input = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
        
        if self.is_top:
            # Handling the top layer differently if it includes labels.
            # Assuming the last n_labels of the visible layer are softmax units for labels.
            data_input = total_input[:, :-self.n_labels]
            label_input = total_input[:, -self.n_labels:]
            
            # Binary data part
            p_data_given_h = sigmoid(data_input)
            v_data = np.random.binomial(n=1, p=p_data_given_h)
            
            # Label part with softmax
            p_labels_given_h = softmax(label_input)
            v_labels = np.array([np.random.choice(self.n_labels, p=p_labels_given_h[i]) for i in range(hidden_minibatch.shape[0])])
            
            # Concatenate data and label parts
            p_v_given_h = np.concatenate((p_data_given_h, p_labels_given_h), axis=1)
            v_samples = np.concatenate((v_data, np.eye(self.n_labels)[v_labels]), axis=1)  # One-hot encode labels
            
        else:
            # For binary visible units
            p_v_given_h = sigmoid(total_input)
            v_samples = np.random.binomial(n=1, p=p_v_given_h)
        
        return p_v_given_h, v_samples


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        
        return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            pass
            
        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            pass
            
        return np.zeros((n_samples,self.ndim_visible)), np.zeros((n_samples,self.ndim_visible))        
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
