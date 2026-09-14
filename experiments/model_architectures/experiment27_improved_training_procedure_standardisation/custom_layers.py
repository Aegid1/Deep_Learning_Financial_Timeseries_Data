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

        # i_oil_t
        self.W_i_oil = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_oil = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_oil = nn.Parameter(torch.Tensor(hidden_sz))

        # i_curr_t
        self.W_i_curr = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_curr = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_curr = nn.Parameter(torch.Tensor(hidden_sz))

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

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock, volume, rsi, macd, vwap, oil_change, currency_change):
        stock, volume, rsi, macd, vwap, oil_change, currency_change = [
            x.unsqueeze(0) if x.dim() == 2 else x
            for x in (stock, volume, rsi, macd, vwap, oil_change, currency_change)
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
            Oil_t = oil_change[:, t, :]
            Currency_t = currency_change[:, t, :]

            i_s_t = torch.relu(S_t @ self.W_i_s + h_t @ self.U_i_s + self.b_i_s)
            i_v_t = torch.relu(V_t @ self.W_i_v + h_t @ self.U_i_v + self.b_i_v)
            i_r_t = torch.relu(R_t @ self.W_i_r + h_t @ self.U_i_r + self.b_i_r)
            i_m_t = torch.relu(M_t @ self.W_i_m + h_t @ self.U_i_m + self.b_i_m)
            i_ms_t = torch.relu(MS_t @ self.W_i_ms + h_t @ self.U_i_ms + self.b_i_ms)
            i_oil_t = torch.relu(Oil_t @ self.W_i_oil + h_t @ self.U_i_oil + self.b_i_oil)
            i_curr_t = torch.relu(Currency_t @ self.W_i_curr + h_t @ self.U_i_curr + self.b_i_oil)

            S_tilde_t = torch.relu(S_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            V_tilde_t = torch.relu(V_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            R_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            M_tilde_t = torch.relu(M_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            MS_tilde_t = torch.relu(MS_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            Oil_tilde_t = torch.relu(Oil_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            Currency_tilde_t = torch.relu(Currency_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            o_t_1 = torch.relu(S_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_2 = torch.relu(R_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_3 = torch.relu(M_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_4 = torch.relu(MS_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_5 = torch.relu(V_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_6 = torch.relu(Oil_tilde_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_7 = torch.relu(Currency_tilde_t @ self.W_o + h_t @ self.U_o + self.b_o)

            o_t = o_t_1 + o_t_2 + o_t_3 + o_t_4 + o_t_5 + o_t_6 + o_t_7

            #if i have more features or more lstms in the previous step i need more values
            l_s_t = S_tilde_t * i_s_t
            l_r_t = R_tilde_t * i_r_t
            l_m_t = M_tilde_t * i_m_t
            l_ms_t = MS_tilde_t * i_ms_t
            l_v_t = V_tilde_t * i_v_t
            l_oil_t = Oil_tilde_t * i_oil_t
            l_curr_t = Currency_tilde_t * i_curr_t

            #if i have more features or more lstms in the previous step i need more values, but matrix stays the same
            u_s_t = l_s_t @ self.W_a * c_t + self.b_a
            u_r_t = l_r_t @ self.W_a * c_t + self.b_a
            u_m_t = l_m_t @ self.W_a * c_t + self.b_a
            u_ms_t = l_ms_t @ self.W_a * c_t + self.b_a
            u_v_t = l_v_t @ self.W_a * c_t + self.b_a
            u_oil_t = l_oil_t @ self.W_a * c_t + self.b_a
            u_curr_t = l_curr_t @ self.W_a * c_t + self.b_a

            #if i have more features or more lstms in the previous step i need more matrices
            alpha_t = torch.softmax(torch.stack([u_s_t, u_r_t, u_m_t, u_ms_t, u_v_t, u_oil_t, u_curr_t]), dim=0)

            L_t = (
                alpha_t[0, :, :] * l_s_t
                + alpha_t[1, :, :] * l_r_t
                + alpha_t[2, :, :] * l_m_t
                + alpha_t[3, :, :] * l_ms_t
                + alpha_t[4, :, :] * l_v_t
                + alpha_t[5, :, :] * l_oil_t
                + alpha_t[6, :, :] * l_curr_t
            )

            c_t = c_t + L_t
            h_t = o_t * torch.relu(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return h_t, hidden_seq
