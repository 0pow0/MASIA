import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMVE(nn.Module):
    """
    Context-Aware Message Value Estimator using lightweight Multi-Head Attention.

    Predicts the marginal contribution (value) of each agent's message
    by modeling dependencies between messages using self-attention.

    Key properties:
    - Permutation equivariant (no positional encoding)
    - Context-aware (attention captures message dependencies)
    - Lightweight (simpler than full Transformer)

    Architecture:
        Input: Messages [bs, n_agents, message_dim]
        → Message Embedding
        → Multi-Head Self-Attention
        → Value Head (MLP)
        Output: Value estimates [bs, n_agents]
    """

    def __init__(self, message_dim, n_agents, hidden_dim=64, num_heads=4, dropout=0.1):
        """
        Args:
            message_dim: Dimension of each message (state_repre_dim)
            n_agents: Number of agents
            hidden_dim: Hidden dimension for attention (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionMVE, self).__init__()

        self.message_dim = message_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input embedding: message_dim -> hidden_dim
        self.message_embed = nn.Linear(message_dim, hidden_dim)

        # Multi-head attention parameters
        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection after attention
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Value head: predicts scalar value for each agent
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def multi_head_attention(self, x, mask=None):
        """
        Multi-head self-attention.

        Args:
            x: [bs, n_agents, hidden_dim]
            mask: Optional [bs, n_agents] binary mask (1=active, 0=dropped)

        Returns:
            attended: [bs, n_agents, hidden_dim]
        """
        bs, n_agents, hidden_dim = x.shape

        # Linear projections: [bs, n_agents, hidden_dim]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: [bs, n_agents, num_heads, head_dim]
        # Then transpose: [bs, num_heads, n_agents, head_dim]
        Q = Q.view(bs, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, n_agents, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [bs, num_heads, n_agents, n_agents]
        scores = th.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided (mask out dropped agents)
        if mask is not None:
            # mask: [bs, n_agents] → expand to [bs, 1, 1, n_agents]
            # This masks out keys (columns) corresponding to dropped agents
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, n_agents]
            # Set attention to -inf for dropped agents (so softmax → 0)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax over keys (last dimension)
        attn_weights = F.softmax(scores, dim=-1)  # [bs, num_heads, n_agents, n_agents]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [bs, num_heads, n_agents, head_dim]
        attended = th.matmul(attn_weights, V)

        # Reshape back: [bs, num_heads, n_agents, head_dim] → [bs, n_agents, hidden_dim]
        attended = attended.transpose(1, 2).contiguous().view(bs, n_agents, hidden_dim)

        # Output projection
        attended = self.W_o(attended)
        attended = self.dropout(attended)

        return attended

    def forward(self, messages, agent_mask=None):
        """
        Forward pass to predict message values.

        Args:
            messages: Tensor of shape [bs, n_agents, message_dim]
                     Messages from all agents (can contain zero vectors for dropped agents)
            agent_mask: Optional binary mask [bs, n_agents] indicating active agents
                       (1 = active, 0 = dropped/silenced)

        Returns:
            values: Tensor of shape [bs, n_agents]
                   Predicted value (marginal contribution) for each agent's message
        """
        # Embed messages: [bs, n_agents, hidden_dim]
        x = self.message_embed(messages)

        # Apply layer norm
        x = self.layer_norm(x)

        # Multi-head attention to capture context
        # Each agent's representation is updated based on all other agents' messages
        attended = self.multi_head_attention(x, mask=agent_mask)

        # Residual connection (optional, helps with gradient flow)
        x = x + attended

        # Predict value for each agent: [bs, n_agents, 1] → [bs, n_agents]
        values = self.value_head(x).squeeze(-1)

        # No ReLU here - values can be negative (harmful messages have negative value)
        # This makes training easier and preserves information about message quality

        # Zero out values for dropped agents if mask is provided
        if agent_mask is not None:
            values = values * agent_mask

        return values

    def compute_loss(self, predicted_values, target_values, agent_mask=None):
        """
        Compute MSE loss between predicted and target message values.

        Args:
            predicted_values: [bs, n_agents] from MVE forward pass
            target_values: [bs, n_agents] ground truth labels
            agent_mask: Optional [bs, n_agents] mask to ignore dropped agents

        Returns:
            loss: Scalar MSE loss
        """
        # Compute squared error
        squared_error = (predicted_values - target_values) ** 2

        # Apply mask if provided (only compute loss for active agents)
        if agent_mask is not None:
            squared_error = squared_error * agent_mask
            # Average over active agents only
            loss = squared_error.sum() / (agent_mask.sum() + 1e-8)
        else:
            # Average over all agents
            loss = squared_error.mean()

        return loss
