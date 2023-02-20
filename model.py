# -*- coding: utf-8 -*-
import torch
import copy
import pickle
import sys
import os
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
# import symbolic constant

from utils import Abnormal_value, MODEL_SAV_PATH, MAX_TO_KEEP

from geomloss import SamplesLoss

from torch.distributions.exponential import Exponential
from torch.distributions.gamma import Gamma
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal

from torch.distributions.kl import kl_divergence
# ====================================================================================================================================================
# Model definition
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class Inference_network(nn.Module):
    def __init__(self,
                 n_topics,
                 n_hidden,
                 vocabulary_size,
                 device,
                 dropout_prob=0.8,
                 inf_nonlinearity='tanh',
                 distribution="Gaussian"):
        super(Inference_network, self).__init__()
        # Inference network for user topic proportion process
        # ---------------------------------------------------------------------------------
        # Critical parameter record
        self.device = device
        self.n_topics = n_topics
        self.distribution = distribution
        # Implement inference network structure
        self.dropout = nn.Dropout(dropout_prob)
        if inf_nonlinearity=='tanh':
            self.MLP_act = nn.Tanh()    
        else:
            self.MPL_act = nn.ReLU()
        self.MLP_linear = nn.Linear(vocabulary_size, n_hidden)
        torch.nn.init.xavier_uniform_(self.MLP_linear.weight.data, gain=1)
        self.batch_norm_alpha = nn.BatchNorm1d(n_topics)
        self.loc = nn.Linear(n_hidden, n_topics)
        torch.nn.init.xavier_uniform_(self.loc.weight.data, gain=1)
        self.log_scale = nn.Linear(n_hidden, n_topics)
        self.log_scale.weight.data.fill_(0.0)
        self.log_scale.bias.data.fill_(0.0)
        self.con_sftpls = nn.Softplus()
        self.scale_sftpls = nn.Softplus()
        self.SD = SamplesLoss(loss='sinkhorn', p=2, backend="tensorized")

    def KL_Gaussian(self, mu, logsigma):
        return -0.5 * torch.sum(1 - torch.pow(mu, 2) + 2 * logsigma - torch.exp(2*logsigma), 1)

    def Prior_samples(self, batch_size, n_dim):
        d = LogNormal(loc=torch.zeros(n_dim), scale=torch.ones(n_dim))
        return d.sample([batch_size]).to(self.device)

    def Sinkhorn_divergence(self, alpha, beta):
        alpha_size = alpha.size()
        beta_size = beta.size()
        n_dim_alpha = alpha_size[-1]
        n_dim_beta = beta_size[-1]
        assert n_dim_beta == n_dim_alpha, "Unequal vector size."
        return self.SD(alpha, beta)

    def LogNormal_Reparameterization(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps = torch.normal(mean=torch.zeros_like(mu))
        return torch.exp(torch.mul(eps, std).add_(mu))

    def Gamma_Re(self, concentration, rate):
        return Gamma(concentration, rate).rsample()

    def forward(self, in_vec):
        enc_vec = self.dropout(self.MLP_act(self.MLP_linear(in_vec)))
        concentration = self.con_sftpls(
            self.batch_norm_alpha(self.loc(enc_vec)))
        rate = 1.0 + self.scale_sftpls(self.log_scale(enc_vec))
        alpha = self.Gamma_Re(concentration, rate).unsqueeze(
            1)  # batch_size, 1, n_topics
        batch_size = alpha.size()[0]
        beta = self.Prior_samples(batch_size, self.n_topics).unsqueeze(
            1)  # batch_size, 1, n_topics
        assert alpha.size() == beta.size(), "Unequal size of 2 distributions."
        SD = self.Sinkhorn_divergence(alpha, beta)
        return SD, alpha.squeeze()

# ---------------------------------------------------------------------------------------------------------------------------------------------------

class Generative_model(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 n_topics,
                 topic_embeddings_size,
                 device,
                 topk,
                 wordembedding_mat=None):  # Must be equal to the one in user_network
        super(Generative_model, self).__init__()
        # Record critical paramters
        self.topk = topk
        self.device = device
        self.n_topics = n_topics
        self.vocabulary_size = vocabulary_size
        self.topic_embeddings_size = topic_embeddings_size
        # Component for document-topic-word process:
        self.theta_softmax = nn.Softmax(dim=-1)
        self.beta_softmax = nn.Softmax(dim=1)
        # self.fweight=fweight()
        self.Sinkhorn = SamplesLoss(
            loss="sinkhorn", p=2, diameter=1.0, backend="tensorized")
        # ========================================================================================
        # topic_embeddings_mat & word_embeddings_mat didnt registered as model parameter
        # ========================================================================================
        topic_embedding_mat = torch.Tensor(topic_embeddings_size, n_topics)
        # Orthogonal initialization
        torch.nn.init.orthogonal_(topic_embedding_mat)
        self.register_parameter('topic_embeddings_mat', Parameter(topic_embedding_mat.t() ) )
        if wordembedding_mat == None:
            word_embeddings_mat = Parameter(torch.Tensor(
                topic_embeddings_size, vocabulary_size))
            torch.nn.init.orthogonal_(word_embeddings_mat.data, gain=1)
            self.register_parameter('word_embeddings_mat', word_embeddings_mat)
        else:
            self.word_embeddings_mat = torch.Tensor(
                wordembedding_mat).to(self.device).t()
            self.word_embeddings_mat.requires_grad = False

    def Reparameterization(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps = torch.normal(mean=torch.zeros_like(mu))
        return torch.mul(eps, std).add_(mu)

    def Get_Beta(self):
        return self.beta

    def Get_Topic_vec(self):
        return self.topic_embeddings_mat

    def Top_k_value_filter_norm(self, input_mat, k=10):
        # Get top-k topic word distribution probability sparse matrix
        vec_length = len(input_mat)
        topk = input_mat.topk(dim=-1, k=k)
        vec_list = []
        for i in range(vec_length):
            filtered_vec = torch.sparse_coo_tensor(topk.indices[i].unsqueeze(
                0), topk.values[i], [self.vocabulary_size], requires_grad=True).to_dense()
            vec_list.append(filtered_vec/filtered_vec.sum(-1))
        return torch.stack(vec_list)  # beta_mat like sparse tensor

    def STDR_Calculate(self, k=10):
        sparse_topic_mat = self.Top_k_value_filter_norm(self.beta, k)
        topic_vec_list = sparse_topic_mat.unbind(0)  # (n_topics, n_dim)
        n_topics = self.n_topics
        STDR_sum = 0.0
        for vec in topic_vec_list:
            x = vec.expand(n_topics, len(vec)).unsqueeze(1)
            y = sparse_topic_mat.unsqueeze(1)
            STDR_sum += self.Sinkhorn(x, y).sum() / n_topics
        Abnormal_value(STDR_sum, "STDR calculation")
        return STDR_sum/n_topics

    # Working flow of generative model
    def forward(self, alpha, doc_bow):
        # Sampling from variational parameters
        theta = self.theta_softmax(alpha)
        wt = torch.mm(self.topic_embeddings_mat, self.word_embeddings_mat)
        self.beta = self.beta_softmax(wt)
        logits = torch.log(torch.mm(theta, self.beta))
        Re = - torch.sum(torch.mul(logits, doc_bow), -1)
        Abnormal_value(Re, "Reconstruction loss")
        STDR = self.STDR_Calculate(k=self.topk)
        Abnormal_value(STDR, "Regularization")
        return Re, STDR, theta
# ====================================================================================================================================================

class ASTM(nn.Module):
    def __init__(self,
                 n_topics,
                 vocabulary_size,
                 n_hidden,
                 dropout_prob,
                 embeddings_size,
                 coel,
                 coea,
                 device,
                 topk,
                 inf_nonlinearity='tanh',
                 alternative_epoch=10,
                 wordembedding_mat=None):
        super(ASTM, self).__init__()
        self.device = device
        self.n_topics = n_topics
        self.coel = coel
        self.coea = coea
        # Initialization for Inference network
        self.vocabulary_size = vocabulary_size
        # Encoder
        self.inf_net = Inference_network(vocabulary_size=vocabulary_size,
                                         n_hidden=n_hidden,
                                         dropout_prob=dropout_prob,
                                         n_topics=n_topics,
                                         inf_nonlinearity=inf_nonlinearity,
                                         device=device)
        # Decoder
        self.gen_model = Generative_model(vocabulary_size=vocabulary_size,
                                          n_topics=n_topics,
                                          topic_embeddings_size=embeddings_size,
                                          wordembedding_mat=wordembedding_mat,
                                          topk=topk,
                                          device=device)

    def forward(self, doc_bow):
        # Invoke Inference network working flow
        SD, alpha = self.inf_net(doc_bow)
        # Invoke Generative model working flow
        Re, STDR, theta = self.gen_model(alpha,
                                         doc_bow)
        loss = Re + self.coel * SD - self.coea * STDR
        return loss, Re, SD, STDR, theta

    def Get_Beta(self):
        return self.gen_model.Get_Beta()
# =================================================================================================================

    def train_model(self,
                    train_dataloader,
                    dev_dataloader,
                    test_dataloader,
                    id2word,
                    learning_rate=1e-4,
                    batch_size=64,
                    training_epoch=1000,
                    alternative_epoch=10):
        inf_optim = optim.Adam(self.inf_net.parameters(), lr=learning_rate)
        gen_optim = optim.Adam(self.gen_model.parameters(), lr=learning_rate)
        min_loss = float('inf')
        epoch_trend = []
        SD_trend = []
        STDR_trend = []
        tu_trend = []
        no_decent_cnt = 0
        # word2id, id2word, vocabulary = ReadDictionary(args.data_path + 'vocab.new')
        # ----------------------------------------------------------------------------------------------------------
        # If previous checkpoint files exist, load the pretrain paramter dictionary from them.
        ckpt = 0
        if os.path.exists(MODEL_SAV_PATH):
            ckpt_list = os.listdir(MODEL_SAV_PATH)
            if len(ckpt_list) > 0:
                for ckpt_f in ckpt_list:
                    current_ckpt = int(ckpt_f.split('-')[1].split('.')[0])
                    if current_ckpt > ckpt:
                        ckpt = current_ckpt
                self.load_state_dict(torch.load(
                    MODEL_SAV_PATH + "model_parameters_epoch-"+str(ckpt).zfill(3)+".pkl"))
                record_list = pickle.load(open('record_list.bin', 'rb'))
                epoch_trend = record_list[0]
                SD_trend = record_list[1]
                STDR_trend = record_list[2]
                tu_trend = record_list[3]
        else:
            os.makedirs(MODEL_SAV_PATH)
        self.to(self.device)
        # ----------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # Main training epoch control
        for epoch in range(ckpt, training_epoch):
            self.train()
            for mode, optimizer in enumerate([gen_optim, inf_optim]):
                if mode == 1:
                    optim_mode = "Updating Encoder parameters"
                else:
                    optim_mode = "Updating Decoder parameters"
                # Alternative training control for wake-sleep algorithm
                for sub_epoch in range(alternative_epoch):
                    # ***Epoch start
                    loss_sum = 0.0
                    SD_sum = 0.0
                    training_word_count = 0
                    doc_count = 0
                    for batch_idx, (doc_bow, label_vec, word_count) in enumerate(train_dataloader):
                        # Input data : vec, label_vec, word_cnt
                        # batch-step
                        data_size = len(doc_bow)
                        doc_bow = doc_bow.to(self.device)
                        # Prepare for model optimization
                        optimizer.zero_grad()
                        loss, Re, SD, STDR, theta = self(doc_bow)
                        loss.backward(torch.ones_like(doc_bow.sum(1)))
                        optimizer.step()
                        loss_sum += torch.sum(loss.detach()).cpu()
                        SD_sum += torch.sum(SD).detach().cpu() / data_size
                        training_word_count += torch.sum(word_count)
                        doc_count += data_size
                    # epoch end
                    SDc = torch.div(SD_sum, len(train_dataloader))
                    loss_value = torch.div(loss_sum, len(train_dataloader))
                    # .to("cpu")
                    print('| Total epoch %3d | Training phase %2d | %s | Loss: %4.3f | SD: %3.3f | STDR: %2.3f' % (epoch,
                                                                                                                   sub_epoch+1,
                                                                                                                   optim_mode,
                                                                                                                   loss_value,
                                                                                                                   SDc,
                                                                                                                   STDR))
            # ==========================================================================================================
            # Evaluating model on dev_dataset
            self.eval()
            dev_loss_sum = 0.0
            SD_sum = 0.0
            test_word_count = 0
            doc_count = 0
            for batch_idx, (doc_bow, label_vec, word_count) in enumerate(dev_dataloader):
                # Input data : vec, label_vec, word_cnt
                data_size = len(doc_bow)
                doc_bow = doc_bow.to(self.device)
                # Model forward function
                with torch.no_grad():
                    loss, Re, SD, STDR, theta = self(doc_bow)
                # Computing evaluation metrics
                dev_loss_sum += torch.sum(loss).detach().cpu()
                SD_sum += torch.sum(SD).detach().cpu() / data_size
                test_word_count += torch.sum(word_count)
                doc_count += data_size
            SDc = torch.div(SD_sum, len(dev_dataloader))
            dev_loss_value = torch.div(dev_loss_sum, len(dev_dataloader))
            print('### Dev phase epoch %3d ## | Loss: %4.3f | SD: %3.3f | STDR: %2.3f' % (epoch,
                                                                                          dev_loss_value,
                                                                                          SDc,
                                                                                          STDR))
            # -----------------------------------------------------------------------------------------------------
            # Evaluating model on test dataset
            test_loss_sum = 0.0
            SD_sum = 0.0
            test_word_count = 0
            doc_count = 0
            for batch_idx, (doc_bow, label_vec, word_count) in enumerate(test_dataloader):
                # Input data : vec, label_vec, word_cnt
                data_size = len(doc_bow)
                doc_bow = doc_bow.to(self.device)
                # Model forward function
                with torch.no_grad():
                    loss, Re, SD, STDR, theta = self(doc_bow)
                # Computing evaluation metrics
                test_loss_sum += torch.sum(loss).detach().cpu()
                SD_sum += torch.sum(SD).detach().cpu() / data_size
                test_word_count += torch.sum(word_count)
                doc_count += data_size
            SDc = torch.div(SD_sum, len(test_dataloader))
            test_loss_value = torch.div(test_loss_sum, len(test_dataloader))
            print('*** Test phase epoch %3d *** | Loss: %4.3f | SD: %3.3f | STDR: %2.3f' % (epoch,
                                                                                            test_loss_value,
                                                                                            SDc,
                                                                                            STDR))
            # ----------------------------------------------------------------------------
            # Record training statics
            epoch_trend.append(epoch)
            SD_trend.append(SDc)
            STDR_trend.append(STDR)
            # Export topic word file
            print("*** Exporting topic word file... ...***")
            beta = self.Get_Beta().detach()
            tu_out = topic_coherence_file_export(
                beta.cpu().numpy(), id2word, idx=epoch, topn=100)
            tu_trend.append(tu_out)
            if dev_loss_value < min_loss:
                min_loss = dev_loss_value
                no_decent_cnt = 0
                with torch.no_grad():
                    # Save model parameters
                    torch.save(self.state_dict(), MODEL_SAV_PATH +
                               "model_parameters_epoch-"+str(epoch).zfill(3)+".pkl")
                    pickle.dump([epoch_trend, SD_trend, STDR_trend,
                                 tu_trend], open('record_list.bin', 'wb'))
                    ckpt_tmp_list = os.listdir(MODEL_SAV_PATH)
                    if len(ckpt_tmp_list) > MAX_TO_KEEP:
                        os.remove(MODEL_SAV_PATH + ckpt_tmp_list[0])
                    pickle.dump(self.gen_model.topic_embeddings_mat.detach(
                    ).cpu().numpy(), open('topic_vec.bin', 'wb'))
            else:
                no_decent_cnt += 1
            if no_decent_cnt > 20:
                return epoch_trend, SD_trend, STDR_trend, tu_trend
        return epoch_trend, SD_trend, STDR_trend, tu_trend
# ====================================================================================================================================================

def topic_coherence_file_export(topicmat, id2word, idx, topn=10):
    # Input topicmat need to be a numpy ndarray
    outputfile = "topic-"+str(idx).zfill(3) + ".txt"
    f = open(outputfile, 'w', encoding='utf-8')
    topic_word_set_list = []
    for topic in topicmat:
        topic_word_list = []
        wordprob_list = []
        word_cnt = 1
        tmp_list = []
        # Build word, probability tuple for each topic
        for index, value in enumerate(topic):
            tmp_list.append((index, value))
        # Decently sort the word according to its probability
        sorted_list = sorted(tmp_list, key=lambda s: s[:][1], reverse=True)
        for pair in sorted_list:
            if word_cnt > topn:
                break
            token = int(pair[0])
            prob = float(pair[1])
            if token not in id2word.keys():
                print(token)
                sys.exit(0)
            assert isinstance(
                id2word[token], str), "Type error on id2word object."
            topic_word_list.append(id2word[token])
            wordprob_list.append((id2word[token], prob))
            word_cnt += 1
        topic_word_set_list.append(topic_word_list)
        f.write(' '.join(topic_word_list)+'\n')
    topic_n = len(topic_word_set_list)
    tu = 0.0
    for topic_i in topic_word_set_list:
        tu_k = 0.0
        for word in topic_i:
            cnt = 0.0
            for topic_j in topic_word_set_list:
                if word in topic_j:
                    cnt += 1.0
            tu_k += 1.0/cnt
        tu_k = tu_k / len(topic_i)
        tu += tu_k
    tu_out = tu/(1.0*topic_n)
    print(" ### Topic uniqueness: %5f " % (tu_out))
    pickle.dump(topic_word_set_list, open(
        "topic_word_distribution-"+str(idx).zfill(3)+".bin", 'wb'))
    f.close()
    return tu_out
