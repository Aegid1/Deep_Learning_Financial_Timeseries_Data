import torch
import torch.nn as nn
import math

class MultiInputLSTMLessLSTMs(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # i_s_t
        self.W_i_s = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_s = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_s = nn.Parameter(torch.Tensor(hidden_sz))

        # i_r_t
        self.W_i_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_r = nn.Parameter(torch.Tensor(hidden_sz))

        # i_sti_t (short_term_impact)
        self.W_i_sti = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_sti = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_sti = nn.Parameter(torch.Tensor(hidden_sz))

        # i_lti_t (long_term_impact)
        self.W_i_lti = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_lti = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_lti = nn.Parameter(torch.Tensor(hidden_sz))

        # i_m_t
        self.W_i_m = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_m = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_m = nn.Parameter(torch.Tensor(hidden_sz))

        # i_ms_t
        self.W_i_ms = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_ms = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_ms = nn.Parameter(torch.Tensor(hidden_sz))

        # i_sm_t
        self.W_i_sm = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_sm = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_sm = nn.Parameter(torch.Tensor(hidden_sz))

        # i_em_t
        self.W_i_em = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_em = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_em = nn.Parameter(torch.Tensor(hidden_sz))

        # i_u_t
        self.W_i_u = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_u = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_u = nn.Parameter(torch.Tensor(hidden_sz))

        # i_mb_t
        self.W_i_mb = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_mb = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_mb = nn.Parameter(torch.Tensor(hidden_sz))

        # i_l_t
        self.W_i_l = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_l = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_l = nn.Parameter(torch.Tensor(hidden_sz))

        # o_l_t
        self.W_o_l = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o_l = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o_l = nn.Parameter(torch.Tensor(hidden_sz))

        # vwap_l_t
        self.W_vwap_l = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_vwap_l = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_vwap_l = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

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

        # forget gate weights (one per stream, shared bias)
        self.W_f_s = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_s = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_m = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_m = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_vwap = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_vwap = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_sti = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_sti = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.W_f_lti = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f_lti = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock, rsi, macd, vwap, short_term_impact, long_term_impact):
        bs, seq_sz, _ = stock.shape
        hidden_seq = []
        #initialize start with zeros
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(stock.device),
            torch.zeros(bs, self.hidden_size).to(stock.device),
        )
        for t in range(seq_sz):
            S_t = stock[:, t, :]
            R_t = rsi[:, t, :]
            M_t = macd[:, t, :]
            Vwap_t = vwap[:, t, :]
            STI_t = short_term_impact[:, t, :]
            LTI_t = long_term_impact[:, t, :]

            # input gates - bounded to [0,1], like a real LSTM gate (was unbounded relu before)
            i_s_t = torch.sigmoid(S_t @ self.W_i_s + h_t @ self.U_i_s + self.b_i_s)
            i_r_t = torch.sigmoid(R_t @ self.W_i_r + h_t @ self.U_i_r + self.b_i_r)
            i_m_t = torch.sigmoid(M_t @ self.W_i_m + h_t @ self.U_i_m + self.b_i_m)
            i_vwap_t = torch.sigmoid(Vwap_t @ self.W_vwap_l + h_t @ self.U_vwap_l + self.b_vwap_l)
            i_sti_t = torch.sigmoid(STI_t @ self.W_i_sti + h_t @ self.U_i_sti + self.b_i_sti)
            i_lti_t = torch.sigmoid(LTI_t @ self.W_i_lti + h_t @ self.U_i_lti + self.b_i_lti)

            # candidate values - bounded to [-1,1] via tanh (was unbounded relu before)
            S_tilde_t = torch.tanh(S_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            R_tilde_t = torch.tanh(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            M_tilde_t = torch.tanh(M_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            Vwap_tilde_t = torch.tanh(Vwap_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            STI_tilde_t = torch.tanh(STI_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            LTI_tilde_t = torch.tanh(LTI_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            # output gate - averaged over streams then squashed to [0,1] (was an unbounded relu sum before)
            o_t_1 = S_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t_2 = R_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t_3 = M_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t_4 = Vwap_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t_5 = STI_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t_6 = LTI_t @ self.W_o + h_t @ self.U_o + self.b_o
            o_t = torch.sigmoid((o_t_1 + o_t_2 + o_t_3 + o_t_4 + o_t_5 + o_t_6) / 6.0)

            # forget gate - decides how much of the previous cell state to keep
            f_s_raw = S_t @ self.W_f_s + h_t @ self.U_f_s
            f_r_raw = R_t @ self.W_f_r + h_t @ self.U_f_r
            f_m_raw = M_t @ self.W_f_m + h_t @ self.U_f_m
            f_vwap_raw = Vwap_t @ self.W_f_vwap + h_t @ self.U_f_vwap
            f_sti_raw = STI_t @ self.W_f_sti + h_t @ self.U_f_sti
            f_lti_raw = LTI_t @ self.W_f_lti + h_t @ self.U_f_lti
            f_t = torch.sigmoid((f_s_raw + f_r_raw + f_m_raw + f_vwap_raw + f_sti_raw + f_lti_raw) / 6.0 + self.b_f)

            l_s_t = S_tilde_t * i_s_t
            l_r_t = R_tilde_t * i_r_t
            l_m_t = M_tilde_t * i_m_t
            l_vwap_t = Vwap_tilde_t * i_vwap_t
            l_sti_t = STI_tilde_t * i_sti_t
            l_lti_t = LTI_tilde_t * i_lti_t

            # attention pre-activations feed straight into softmax, unclamped (relu here would
            # collapse every negative pre-activation to the same 0, distorting the attention weights)
            u_s_t = l_s_t @ self.W_a * c_t + self.b_a
            u_r_t = l_r_t @ self.W_a * c_t + self.b_a
            u_m_t = l_m_t @ self.W_a * c_t + self.b_a
            u_vwap_t = l_vwap_t @ self.W_a * c_t + self.b_a
            u_sti_t = l_sti_t @ self.W_a * c_t + self.b_a
            u_lti_t = l_lti_t @ self.W_a * c_t + self.b_a

            alpha_t = torch.softmax(torch.stack([u_s_t, u_r_t, u_m_t, u_vwap_t, u_sti_t, u_lti_t]), dim=0)
            L_t = (
                alpha_t[0, :, :] * l_s_t
                + alpha_t[1, :, :] * l_r_t
                + alpha_t[2, :, :] * l_m_t
                + alpha_t[3, :, :] * l_vwap_t
                + alpha_t[4, :, :] * l_sti_t
                + alpha_t[5, :, :] * l_lti_t
            )

            # forget gate replaces pure accumulation, tanh bounds the cell state passed to h_t
            c_t = f_t * c_t + L_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return h_t, hidden_seq
