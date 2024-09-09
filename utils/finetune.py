import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gpn.model
from transformers import AutoModel,AutoTokenizer
#from rinalmo.pretrained import get_pretrained_model



class GPNModel(nn.Module):
    def __init__(self, model_name='songlab/gpn-brassicales', unfreeze_last_n_layers = 1):
        super(GPNModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # self.model.encoder = torch.compile(self.model.encoder)
        self.model = torch.compile(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained('songlab/gpn-brassicales')
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
       
    def tokenize_function(self, sequences):
        return self.tokenizer(sequences, return_tensors="pt", return_attention_mask=False, 
                            return_token_type_ids=False)

    def forward(self, sequences, device='cuda'):
        self.model = self.model.to(device)
        output_embeddings = []
        for i in sequences:
            input_ids = self.tokenize_function(i)["input_ids"].to(device)
            embedding = self.model(input_ids=input_ids).last_hidden_state.squeeze()
            output_embeddings.append(embedding)

        return output_embeddings

    def parameters(self, **kwargs):
        parameter_list = []
        for name, param in self.model.named_parameters():
            for i in range(self.unfreeze_last_n_layers):
                if f'mod.encoder.{24-i}' in name:
                    parameter_list.append(param)
        return parameter_list
    
    def train(self, train = True):
        
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for i in range(self.unfreeze_last_n_layers):

            for name, param in self.model.named_parameters():

                if f'mod.encoder.{24-i}' in name:
                    param.requires_grad = train


    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def verbose(self):
        print("GPN Model: songlab/gpn-brassicales with total layers 24")
        print(f"Number of layers trainable: {self.unfreeze_last_n_layers}") 

class ESMModel(nn.Module):
    available_models = {
        "esm2_t48_15B_UR50D": 5120,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t6_8M_UR50D": 320
    }

    def __init__(self, model_name="esm2_t6_8M_UR50D", unfreeze_last_n_layers=1):
        super(ESMModel, self).__init__()
        
        if model_name not in self.available_models:
            raise ValueError(f"Model name '{model_name}' is not valid. Choose from: {', '.join(self.available_models.keys())}")

        self.model_name = model_name
        self.embedding_dim = self.available_models[model_name]
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.batch_converter = self.alphabet.get_batch_converter()
        self.num_layers = int(model_name.split('_')[1][1:])  # Extract number of layers from model name
        self.freeze_layers = self.num_layers - self.unfreeze_last_n_layers
        self.unfrozen_layers = list(range(self.freeze_layers, self.num_layers))

    def forward(self, sequences, device="cuda"):
        self.model = self.model.to(device)
        _, _, batch_tokens = self.batch_converter(sequences)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.model(batch_tokens.to(device), repr_layers=[self.num_layers], return_contacts=False)
        token_representations = results["representations"][self.num_layers]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1])
        return sequence_representations

    def parameters(self, **kwargs):
        # Return parameters for all unfrozen layers
        return [p for i in self.unfrozen_layers for p in self.model.layers[i].parameters()]

    def train(self, mode=True):
        #unfrozen_layers = list(range(self.freeze_layers, self.num_layers))
        for i in range(self.num_layers):
            for param in self.model.layers[i].parameters():
                if i in self.unfrozen_layers:
                    param.requires_grad = mode
                else:
                    param.requires_grad = False    

        # Freeze other stuff
        if hasattr(self.model, 'contact_head'):

            for param in self.model.contact_head.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'lm_head'):

            for param in self.model.lm_head.parameters():
                param.requires_grad = False    

        if hasattr(self.model, 'emb_layer_norm_after'):
            for param in self.model.emb_layer_norm_after.parameters():
                param.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))

    def verbose(self):
        print(f"ESM Model:{self.model_name} with total layers: {self.num_layers}")
        print(f"Freezing the first {self.freeze_layers} layers and unfreezing the last {self.unfreeze_last_n_layers} layers.")
        print(f"Trainable layers: {self.unfrozen_layers}")


# class RiNALMoModel(nn.Module):

#     available_models = {
#         "giga-v1": 1280  
#     }

#     def __init__(self, model_name="giga-v1", device="cuda", padding_idx=0, unfreeze_last_n_layers=1):
#         super(RiNALMoModel, self).__init__()
        
#         if model_name not in self.available_models:
#             raise ValueError(f"Model name '{model_name}' is not valid. Choose from: {', '.join(self.available_models.keys())}")

#         self.model_name = model_name
#         self.embedding_dim = self.available_models[model_name]
#         self.model, self.alphabet = get_pretrained_model(model_name=model_name)
#         self.model = self.model.to(device)
#         self.num_layers = len(self.model.transformer.blocks)
#        # print(f"Number of layers in the model: {self.num_layers}")
#         self.unfreeze_last_n_layers = unfreeze_last_n_layers
#         self.unfrozen_layers = list(range((self.num_layers - self.unfreeze_last_n_layers), self.num_layers))
#         self.device = device
#         self.padding_idx = padding_idx 
        

#     def forward(self, sequences, device="cuda"):
#         self.model = self.model.to(device)

        
#         tokenized_sequences = [torch.tensor(self.alphabet.batch_tokenize([seq]), dtype=torch.int64).squeeze(0) for seq in sequences]
#         original_lengths = [len(seq) for seq in tokenized_sequences]
#         padded_tokens = pad_sequence(tokenized_sequences, batch_first=True, padding_value=self.padding_idx).to(device)

#         with torch.cuda.amp.autocast():
#             outputs = self.model(padded_tokens)
#             representations = outputs["representation"]
#             unpadded_representations = []
#             for i, length in enumerate(original_lengths):
#                 # Remove CLS and SEP tokens, and unpad the representations
#                 unpadded_representation = representations[i][1:length-1].cpu()
#                 unpadded_representations.append(unpadded_representation)

#         return unpadded_representations

#     def parameters(self, **kwargs):
#         # Return parameters for all unfrozen layers
#         freeze_layers = self.num_layers - self.unfreeze_last_n_layers
#         # print(f"Freezing the first {freeze_layers} layers and unfreezing the last {self.unfreeze_last_n_layers} layers.")
#         # print(f"Unfrozen layers: {self.unfrozen_layers}")
#         return [p for i in self.unfrozen_layers for p in self.model.transformer.blocks[i].parameters()]

#     def train(self, train=True):
       
#         if self.unfreeze_last_n_layers < 1 or self.unfreeze_last_n_layers > self.num_layers:
#             raise ValueError(f"unfreeze_last_n_layers must be between 1 and {self.num_layers}.")

    
#         freeze_layers = self.num_layers - self.unfreeze_last_n_layers
        

#         # Freeze all layers except the last 'unfreeze_last_n_layers'
#         for i in range(freeze_layers):
#             self.model.transformer.blocks[i].training = False
#             for param in self.model.transformer.blocks[i].parameters():
#                 param.requires_grad = False

#         # Unfreeze the specified last layers
#         for i in range(freeze_layers, self.num_layers):
#             for param in self.model.transformer.blocks[i].parameters():
#                 param.requires_grad = True

#         # Set the selected layers to train mode
#         for i in self.unfrozen_layers:
#             self.model.transformer.blocks[i].train(train)

#     def save_checkpoint(self, path):
#         torch.save(self.model.state_dict(), path)

#     def load_checkpoint(self, path):
#         self.model.load_state_dict(torch.load(path))

#     def verbose(self):
#         print(f"RiNALMo Model:{self.model_name} with total layers: {self.num_layers}")
#         print(f"Freezing the first {self.num_layers - self.unfreeze_last_n_layers} layers and unfreezing the last {self.unfreeze_last_n_layers} layers.")
        
#         print(f"Trainable Layers: {self.unfrozen_layers}")





