import torch
import torch.nn as nn
import math
from utils.experiment_utils import debug_activations

class MultiInputLSTMLessLSTMs(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # i_s_t
        self.W_i_s = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_s = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_s = nn.Parameter(torch.Tensor(hidden_sz))

        # i_v_t
        self.W_i_v = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_v = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_v = nn.Parameter(torch.Tensor(hidden_sz))

        # i_r_t
        self.W_i_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_r = nn.Parameter(torch.Tensor(hidden_sz))

        # i_m_t
        self.W_i_m = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_m = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_m = nn.Parameter(torch.Tensor(hidden_sz))

        # i_ms_t
        self.W_i_ms = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_ms = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_ms = nn.Parameter(torch.Tensor(hidden_sz))

        # i_em_t
        self.W_i_em = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_em = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_em = nn.Parameter(torch.Tensor(hidden_sz))

        # c_p_t
        self.W_c_p = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c_p = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_p = nn.Parameter(torch.Tensor(hidden_sz))

        # c_n_t
        self.W_c_n = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c_n = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_n = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # a_t
        self.W_a = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_a = nn.Parameter(torch.Tensor(hidden_sz))

        # forget gate weights
        self.W_f_s = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_s = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_v = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_v = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_m = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_m = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_ms = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_ms = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_oil = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_oil = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_curr = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_curr = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        # one global bias
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    #TODO check the initialization and maybe revert it to uniform initialization
    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith('W_'): # Input-to-Hidden weights
                nn.init.xavier_uniform_(param)

            elif name.startswith('U_'): # Hidden-to-Hidden weights
                nn.init.orthogonal_(param)

            elif name.startswith('b_'):
                if name == 'b_f': # biases
                    nn.init.ones_(param) # global forget bias
                else:
                    nn.init.zeros_(param) # all other biases to 0

    def forward(self, stock, volume, rsi, macd, vwap):
        stock, volume, rsi, macd, vwap = [
            x.unsqueeze(0) if x.dim() == 2 else x
            for x in (stock, volume, rsi, macd, vwap)
        ]
        bs, seq_sz, _ = stock.shape
        hidden_seq = []
        #initialize start with zeros
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(stock.device),
            torch.zeros(bs, self.hidden_size).to(stock.device),
        )
        for t in range(seq_sz):
            S_t = stock[:, t, :]
            V_t = volume[:, t, :]
            R_t = rsi[:, t, :]
            M_t = macd[:, t, :]
            MS_t = vwap[:, t, :]

            #if i have more features or more lstms in the previous step i need more matrices
            i_s_t = torch.sigmoid(S_t @ self.W_i_s + h_t @ self.U_i_s + self.b_i_s)
            i_v_t = torch.sigmoid(V_t @ self.W_i_v + h_t @ self.U_i_v + self.b_i_v)
            i_r_t = torch.sigmoid(R_t @ self.W_i_r + h_t @ self.U_i_r + self.b_i_r)
            i_m_t = torch.sigmoid(M_t @ self.W_i_m + h_t @ self.U_i_m + self.b_i_m)
            i_ms_t = torch.sigmoid(MS_t @ self.W_i_ms + h_t @ self.U_i_ms + self.b_i_ms)

            #if I have more features or more lstms in the previous step I need more matrices
            S_tilde_t = torch.tanh(S_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            V_tilde_t = torch.tanh(V_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            R_tilde_t = torch.tanh(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            M_tilde_t = torch.tanh(M_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            MS_tilde_t = torch.tanh(MS_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            #add more outputs
            o_t_1 = torch.sigmoid(S_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_2 = torch.sigmoid(R_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_3 = torch.sigmoid(M_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_4 = torch.sigmoid(MS_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_5 = torch.sigmoid(V_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t = torch.sigmoid((o_t_1 + o_t_2 + o_t_3 + o_t_4 + o_t_5)/5.0)

            #if i have more features or more lstms in the previous step i need more values
            l_s_t = S_tilde_t * i_s_t
            l_r_t = R_tilde_t * i_r_t
            l_m_t = M_tilde_t * i_m_t
            l_ms_t = MS_tilde_t * i_ms_t
            l_v_t = V_tilde_t * i_v_t

            #if i have more features or more lstms in the previous step i need more values, but matrix stays the same
            u_s_t = l_s_t @ self.W_a * c_t + self.b_a
            u_r_t = l_r_t @ self.W_a * c_t + self.b_a
            u_m_t = l_m_t @ self.W_a * c_t + self.b_a
            u_ms_t = l_ms_t @ self.W_a * c_t + self.b_a
            u_v_t = l_v_t @ self.W_a * c_t + self.b_a

            #if i have more features or more lstms in the previous step i need more matrices
            alpha_t = torch.softmax(torch.stack([u_s_t, u_r_t, u_m_t, u_ms_t, u_v_t]), dim=0)

            L_t = (
                alpha_t[0, :, :] * l_s_t
                + alpha_t[1, :, :] * l_r_t
                + alpha_t[2, :, :] * l_m_t
                + alpha_t[3, :, :] * l_ms_t
                + alpha_t[4, :, :] * l_v_t
            )

            # forget gate calculation
            f_s_raw = S_t @ self.W_f_s + h_t @ self.U_f_s
            f_v_raw = V_t @ self.W_f_v + h_t @ self.U_f_v
            f_r_raw = R_t @ self.W_f_r + h_t @ self.U_f_r
            f_m_raw = M_t @ self.W_f_m + h_t @ self.U_f_m
            f_ms_raw = MS_t @ self.W_f_ms + h_t @ self.U_f_ms
            f_t = torch.sigmoid((f_s_raw + f_v_raw + f_r_raw + f_m_raw + f_ms_raw) / 5.0 + self.b_f)

            c_t = f_t * c_t + L_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return h_t, hidden_seq
