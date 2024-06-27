from torch import nn
import torch

from ..network import ScaledDotAttention


class MultiHeadAttention(nn.Module):

    def __init__(
        self, d_model: int, d_k: int, d_v: int, n_heads: int, dropout: float = 0.0
    ):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_m = d_model

        self.weights_q = None
        self.weights_k = None
        self.weights_v = None
        self.attention = None
        self.project = None
        self.dropout = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       -Initialize all weight layers as linear layers                 #
        #       -Initialize the ScaledDotAttention                             #
        #       -Initialize the projection layer as a linear layer             #
        #  Task 13:                                                            #
        #       -Initialize the dropout layer (torch.nn implementation)        #
        #                                                                      #
        # Hints 3:                                                             #
        #       - Instead of initializing several weight layers for each head, #
        #         you can create one large weight matrix. This speed up        #
        #         the forward pass, since we dont have to loop through all     #
        #         heads!                                                       #
        #       - All linear layers should only be a weight without a bias!    #
        ########################################################################

        self.weights_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.attention = ScaledDotAttention(d_k)
        self.project = nn.Linear(d_v * n_heads, d_model, bias=False)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Mask

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - mask: (batch_size, sequence_length_queries, sequence_length_keys)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Pass q,k and v through the linear layer                      #
        #       - Split the last dimensions into n_heads and d_k od d_v        #
        #       - Swap the dimensions so that the shape matches the required   #
        #         input shapes of the ScaledDotAttention layer                 #
        #       - Pass them through the ScaledDotAttention layer               #
        #       - Swap the dimensions of the output back                       #
        #       - Combine the last two dimensions again                        #
        #       - Pass the outputs through the projection layer                #
        #   Task 8:                                                            #
        #       - If a mask is given, add an empty dimension at dim=1          #
        #       - Pass the mask to the ScaledDotAttention layer                #
        #  Task 13:                                                            #
        #       - Add dropout as a final step after the projection layer       #
        #                                                                      #
        # Hints 3:                                                             #
        #       - It helps to write down which dimensions you want to have on  #
        #         paper!                                                       #
        #       - Above the todo, we have already extracted the batch_size and #
        #         the sequence lengths for you!                                #
        #       - Use reshape() to split or combine dimensions                 #
        #       - Use transpose() again to swap dimensions                     #
        # Hints 8:                                                             #
        #       - Use unsqueeze() to add dimensions at the correct location    #
        ########################################################################

        # Pass through linear layer
        q_ = self.weights_q(q)
        k_ = self.weights_k(k)
        v_ = self.weights_v(v)

        # Reshape weights
        q_ = torch.reshape(q_, [*q_.shape[:-1], self.n_heads, self.d_k])
        k_ = torch.reshape(k_, [*k_.shape[:-1], self.n_heads, self.d_k])
        v_ = torch.reshape(v_, [*v_.shape[:-1], self.n_heads, self.d_v])

        # Transpose
        q_ = q_.transpose(1, 2)
        k_ = k_.transpose(1, 2)
        v_ = v_.transpose(1, 2)

        # Pass through attention layer
        outputs = self.attention(q_, k_, v_)
        outputs = outputs.transpose(1, 2)
        outputs = torch.reshape(outputs, [*outputs.shape[:-2], self.n_heads * self.d_v])
        outputs = self.project(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
