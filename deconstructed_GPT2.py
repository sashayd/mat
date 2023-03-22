import torch
import torch.nn as nn
import math

import utils.torch as trc

from aux import mul, concat, get_model, load_mat

#############################


class Rep:
    _follow_attributes =\
        ['_tokenized_sentence_list',
         '_tokenized_sentence_tensor',
         '_am', '_am2', '_token_positions',
         '_layer',
         '_layers_processed',
         '_device',
         '_reduced']

    _dtype = torch.float32
    _dtype_int = torch.int64

    def __init__(self, *args, **kwargs):
        super().__init__()

        for attr in self._follow_attributes:
            setattr(self, attr, None)
        setattr(self, '_device', 'cpu')
        setattr(self, '_layers_processed', 0.)
        self._v = None

        a = args[0]
        if isinstance(a, list):
            self._tokenized_sentence_list = a
        elif isinstance(a, torch.Tensor):
            self._v = a
            if 'follow' in kwargs and kwargs['follow'] is not None:
                for attr in self._follow_attributes:
                    setattr(self, attr, getattr(kwargs['follow'], attr))
        if 'device' in kwargs and kwargs['device'] is not None:
            self._device = kwargs['device']
        if ('token_positions' in kwargs and
            kwargs['token_positions'] is not None):
            self._reduced = False
            self._token_positions = kwargs['token_positions'].to(self._device)

    @staticmethod
    def _pad(vs):
        return torch.nn.utils.rnn.pad_sequence(vs,
                                               batch_first=True,
                                               padding_value=0)

    def to(self, device):
        new = Rep(self.v().to(device), follow=self)
        new._device = device
        for attr in ['_tokenized_sentence_tensor',
                     '_am', '_am2',
                     '_token_positions']:
            if getattr(new, attr) is not None:
                setattr(new, attr, getattr(new, attr).to(device))
        return new

    def batch(self, batch_size=64):
        ts = self.ts()
        if ts is None:
            raise RuntimeError('illegal to batch Rep which '
                               'does not have tokenized sentence list')
        i = 0
        while i < len(ts):
            i1 = min(i + batch_size, len(ts))
            tp = None if self.tp() is None else self.tp()[i:i1]
            batch = Rep(ts[i:i1],
                        token_positions=tp,
                        device=self.device())
            yield batch
            i = i1

    def _create_sent_tensor_and_am(self):
        if self._tokenized_sentence_list is None:
            raise RuntimeError('_tokenized_sentence_list is None!')
        else:
            sents = self._tokenized_sentence_list
            if self._token_positions is None:
                am = [torch.ones((len(sent),), dtype=self._dtype_int)
                      for sent in sents]
            else:
                sents = [sents[i][:self._token_positions[i]+1] for
                         i in range(len(sents))]
                am = [torch.ones((self._token_positions[i]+1,),
                                 dtype=self._dtype_int)
                      for i in range(len(sents))]
            sents = self._pad(sents).to(self._device)
            am = self._pad(am).to(self._device)
            self._tokenized_sentence_tensor = sents
            self._am = am

    def device(self):
        return self._device

    def ts(self):
        return self._tokenized_sentence_list

    def tp(self):
        return self._token_positions

    def st(self):
        if self._tokenized_sentence_tensor is None:
            self._create_sent_tensor_and_am()
        return self._tokenized_sentence_tensor

    def am(self):
        if self._am is None:
            self._create_sent_tensor_and_am()
        return self._am

    def am2(self):
        if self._am2 is None:
            if self._am is None:
                self._create_sent_tensor_and_am()
            am = self._am
            am = am.view(am.shape[0], -1)
            am = am[:, None, None, :]
            am = am.to(dtype=self._dtype)
            am = (1.0 - am) * torch.finfo(self._dtype).min
            self._am2 = am
        return self._am2

    def v(self):
        if self._v is None:
            raise RuntimeError('_v is None!')
        return self._v

    def size(self):
        if self._v is None:
            raise RuntimeError('_v is None!')
        return self._v.size()

    def underwent_emb(self):
        if self._layer is None:
            return False
        else:
            return True

    def layer(self):
        return self._layer

    def set_layer(self, layer):
        self._layer = layer

    def layers_processed(self):
        return self._layers_processed

    def add_processed_layers(self, layers):
        self._layers_processed += layers

    def reduce(self):
        if self.is_reduced() is None or self.is_reduced() is True:
            return self
        if len(self.size()) != 3:
            raise RuntimeError('non-reduced Rep object has size '
                               'of length not 3')
        v_new = trc.select_1_by_0(self.v(), self.tp())
        rep_new = Rep(v_new, follow=self)
        rep_new._reduced = True
        rep_new._am = None
        rep_new._am2 = None
        return rep_new

    def is_reduced(self):
        return self._reduced

    def apply(self, f, new_layer=None, layers_processed=None):
        v_new = f(self.v())
        v_new = Rep(v_new, follow=self)
        if new_layer is not None:
            v_new.set_layer(new_layer)
        if layers_processed is not None:
            v_new.add_processed_layers(layers_processed)
        return v_new

    def softmax(self, **kwargs):
        f = (lambda v: v.softmax(dim=-1))
        return self.apply(f, **kwargs)

    def topk(self, k=10, **kwargs):
        f = (lambda v: v.topk(k=k, dim=-1)[1])
        return self.apply(f, **kwargs)

    def topkvals(self, k=10, **kwargs):
        f = (lambda v: v.topk(k=k, dim=-1)[0])
        return self.apply(f, **kwargs)

    def mul(self, A, **kwargs):
        f = (lambda v: mul(A, v))
        return self.apply(f, **kwargs)

    def bias(self, b, **kwargs):
        f = (lambda v: v + b)
        return self.apply(f, **kwargs)

    def __add__(self, b):
        result = Rep(concat(self.v(), b.v()))
        if self.layer() == b.layer():
            result._layer = self.layer()
        n = self.size()[0]
        m = b.size()[0]
        result._layers_processed =\
            (n * self._layers_processed + m * b._layers_processed) / (n+m)
        result._device = self._device
        if self.is_reduced() == b.is_reduced():
            result._reduced = self.is_reduced()
        return result

#############################


class DeconstructedGPT2(nn.Module):

    def __init__(self, model_name, dataset=None,
                 params=None, model=None, tokenizer=None,
                 device=None):
        super().__init__()

        # this is just for self._one_block,
        # default is False,
        # when computing linear regression
        # we set it to True
        self._no_ln_f = False

        self._model_name = model_name
        self._dataset = dataset
        if params is None:
            self._params = {'mode': 'raw'}
        else:
            self._params = params
        self._E = None
        self._As = {}

        if model is None:
            self._model = get_model(model_name)
            self._device = 'cpu'
        else:
            self._model = model
            self._device = device
        if device is not None:
            self._model.to(device)
            self._device = device

        self._num_of_layers = len(self._model.h)

        self._do_partial_registry = False

        parts_list = ['save_res_1',
                      (1, 'ln_1'),
                      (2, 'attn'),
                      (3, 'res_1'),
                      'save_res_2',
                      (4, 'ln_2'),
                      (5, 'mlp'),
                      (6, 'res_2')]
        self._parts_list = parts_list

    @staticmethod
    def _early_exit_threshold(layer,
                              token_pos,
                              params,
                              plain=False):
        if plain:
            return params['lambda']
        if layer < params['minimal_layer']:
            return 2.
        else:
            result = 0.
            result += 0.9 * params['lambda']
            result += 0.1 * math.exp(
                -params['tau'] *
                token_pos /
                params['N']
                )
            return result

    def params(self):
        return self._params

    def device(self):
        return self._device

    def set_params(self, params):
        self._params = params

    def num_of_layers(self):
        return self._num_of_layers

    def to(self, device):
        self._device = device
        super().to(device)
        if self._E is not None:
            self._E = self._E.to(device)
        for key in self._As:
            if isinstance(self._As[key], list):
                self._As[key] = [c.to(device) for c in self._As[key]]
            else:
                self._As[key] = self._As[key].to(device)

    def _load_mat(self, indices):
        if indices not in self._As:
            self._As[indices] = load_mat(self._model_name,
                                         indices,
                                         dataset=self._dataset,
                                         device=self._device)

    def _emb(self, v, **kwargs):
        result = v.st()

        inputs_embeds = self._model.wte(result)
        position_ids = torch.arange(0, result.size()[-1],
                                    dtype=torch.long,
                                    device=self._device)
        position_embeds = self._model.wpe(position_ids)
        result = inputs_embeds + position_embeds

        result = Rep(result, follow=v)
        result.set_layer(0)

        return result

    def _one_block(self, v,
                   do_partial_registry=False,
                   what_to_mat=None):
        if not v.underwent_emb():
            raise RuntimeError()

        layer = v.layer()
        if not (layer >= 0 and layer < self._num_of_layers):
            raise RuntimeError()

        result = v.v()
        am = v.am2()

        block = self._model.h[layer]

        if (not do_partial_registry) and what_to_mat is None:
            result = block(result, attention_mask=am)[0]
        else:
            if do_partial_registry:
                partial_registry = {}
                partial_registry[0] =\
                    trc.select_1_by_0(result, v.tp())
            for part in self._parts_list:
                if isinstance(part, str) and part.startswith('save_res'):
                    residual = result
                elif isinstance(part, tuple):
                    i = part[0]
                    name = part[1]
                    if what_to_mat is not None and i in what_to_mat:
                        self._load_mat((layer, i-1, i))
                        A = self._As[(layer, i-1, i)]
                        # if there is a bias term
                        if isinstance(A, list):
                            result = mul(A[0], result)
                            result += A[1]
                        else:
                            result = mul(A, result)
                    else:
                        if name == 'attn':
                            result =\
                                getattr(block, name)(result,
                                                     attention_mask=am)[0]
                        elif name.startswith('res'):
                            result = residual + result
                        else:
                            result = getattr(block, name)(result)
                    if do_partial_registry:
                        partial_registry[part[0]] =\
                            trc.select_1_by_0(result, v.tp())

        if do_partial_registry:
            self._partial_registry = partial_registry

        if (layer+1 == self._num_of_layers) and (not self._no_ln_f):
            result = self._model.ln_f(result)

        result = Rep(result, follow=v)
        result.set_layer(layer+1)
        if what_to_mat is None or len(what_to_mat) == 0:
            result.add_processed_layers(1)

        return result

    def _blk(self, v, end):
        result = v

        for i in range(v.layer(), end-1):
            result = self._one_block(result)

        result =\
            self._one_block(result,
                            do_partial_registry=self._do_partial_registry)

        return result

    def _blk_mat(self, v, end):
        result = v

        for i in range(v.layer(), end):
            result = self._one_block(result,
                                     what_to_mat=self._params['what_to_mat'])

        return result

    def _mat(self, v, end):
        start = v.layer()

        if start == end:
            return v

        self._load_mat((start, end))
        A = self._As[(start, end)]

        result = v
        # if there is a bias term
        if isinstance(A, list):
            result = result.mul(A[0], new_layer=end)
            result = result.bias(A[1])
        else:
            result = result.mul(A, new_layer=end)
        if end == self._num_of_layers:
            result = result.apply(self._model.ln_f)
        return result

    def _id(self, v, end):
        result = v
        result = result.apply((lambda x: x), new_layer=end)
        return result

    def _idln(self, v, end):
        result = v
        f = self._model.ln_f if\
            (end == self._num_of_layers and
             v.layer() != self._num_of_layers)\
                else (lambda x: x)
        result = result.apply(f, new_layer=end)
        return result

    def _e(self, v):
        # maybe this detach() is unnecessary and wasteful...
        if self._E is None:
            self._E = self._model.wte.weight.detach()
        E = self._E
        result = v
        result = result.mul(E)
        return result

    def _decision_raw(self, v):
        return 'continue', 'blk', self._num_of_layers

    def _decision_early_exit(self, v):
        if v.is_reduced() is None or v.is_reduced() is True:
            raise RuntimeError()
        if v.size()[0] != 1:
            raise RuntimeError()

        ee_jump_mode = self._params['ee_jump_mode']
        # if early exit jump mode involves blocks, first jump
        # and then reduce, else reduce and then jump
        if ee_jump_mode == 'blk_mat' or ee_jump_mode == 'blk':
            final_layer_result =\
                getattr(self, '_' + ee_jump_mode)(
                    v,
                    self._num_of_layers)
            final_layer_result = final_layer_result.reduce()
        else:
            final_layer_result =\
                getattr(self, '_' + ee_jump_mode)(
                    v.reduce(),
                    self._num_of_layers)
        test = self._e(final_layer_result)
        test = test.softmax().topkvals(k=2)
        test = (test.v()[..., 0] - test.v()[..., 1]).item()
        if test > self._early_exit_threshold(v.layer(),
                                             v.tp().item(),
                                             self._params):
            return 'early_exit', final_layer_result
        else:
            return 'continue', 'blk', v.layer() + 1

    def _decision_jump(self, v):
        jump_layer = self._params['jump_layer']
        jump_mode = self._params['jump_mode']
        if v.layer() < jump_layer:
            return 'continue', 'blk', jump_layer
        else:
            if jump_mode == 'stop':
                return 'stop', None, None
            else:
                return 'continue', jump_mode, self._num_of_layers

    def forward(self, v):
        result = v

        if not result.underwent_emb():
            result = self._emb(result)

        while result.layer() < self._num_of_layers:
            decision =\
                getattr(self,
                        '_decision_' + self._params['mode'])(result)
            if decision[0] == 'continue':
                mode = decision[1]
                end_layer = decision[2]
                result =\
                    getattr(self, '_' + mode)(result, end_layer)
            elif decision[0] == 'early_exit':
                result = decision[1]
                break
            elif decision[0] == 'stop':
                break

        return result

    def forward_detailed(self, v,
                         what_to_return=None, k=10):
        if what_to_return is None:
            what_to_return = set(['output'])

        what_to_return_order = ['output_nonreduced', 'partial_registry',
                                'output', 'output_withE',
                                'output_softmax', 'perplexity', 'topk']

        break_point = max([i for i in range(len(what_to_return_order))
                           if what_to_return_order[i] in what_to_return])

        if 'partial_registry' in what_to_return:
            self._do_partial_registry = True

        return_dict = {}

        with torch.no_grad():
            break_counter = 0
            while break_counter <= break_point:
                if break_counter == 0:
                    result = self.forward(v)
                    if 'output_nonreduced' in what_to_return:
                        return_dict['output_nonreduced'] =\
                            result
                if break_counter == 1:
                    if 'partial_registry' in what_to_return:
                        return_dict['partial_registry'] =\
                            self._partial_registry
                        del self._partial_registry
                if break_counter == 2:
                    if 'perplexity' not in what_to_return:
                        result = result.reduce()
                    if 'output' in what_to_return:
                        return_dict['output'] =\
                            result
                if break_counter == 3:
                    result = self._e(result)
                    if 'output_withE' in what_to_return:
                        return_dict['output_withE'] =\
                            result
                if break_counter == 4:
                    result = result.softmax()
                    if 'output_softmax' in what_to_return:
                        return_dict['output_softmax'] =\
                            result
                if break_counter == 5:
                    if 'perplexity' in what_to_return:
                        perp = trc.select_2_by_01(result.v(),
                                                  v.st().roll(-1, 1))
                        perp = -torch.log(perp)
                        mask = v.am().roll(-1, 1) * v.am()
                        perp = perp * mask
                        perp = perp.sum(dim=1) / mask.sum(dim=1)
                        return_dict['perplexity'] =\
                            perp
                        result = result.reduce()
                if break_counter == 6:
                    result = result.topk(k=k)
                    if 'topk' in what_to_return:
                        return_dict['topk'] =\
                            result
                break_counter += 1

        self._do_partial_registry = False

        return return_dict

    # bh stands for "batched horizontal"
    def forward_detailed_bh(self, v,
                            what_to_return=None, k=10,
                            final_device=None, batch_size=64,
                            instruction_list=None):
        saved_params = self.params()

        if what_to_return is None:
            what_to_return = set(['output'])
        if k is None:
            k = 10
        if batch_size is None:
            batch_size = 64

        if instruction_list is None:
            instruction_list = [(self.params(), 'output')]

        v_batches = list(v.batch(batch_size))

        for params, instruction in instruction_list:
            self.set_params(params)
            wtr =\
                set(['output_nonreduced']) if 'save' in instruction else set()
            wtr = wtr | set(what_to_return)
            outputs = []
            for i in range(len(v_batches)):
                output =\
                    self.forward_detailed(v_batches[i],
                                          what_to_return=wtr,
                                          k=k)
                if 'save' in instruction:
                    v_batches[i] = output['output_nonreduced']
                    output.pop('output_nonreduced')
                if 'output' in instruction:
                    if final_device is not None:
                        for key in output:
                            if (isinstance(output[key], torch.Tensor) or
                                isinstance(output[key], Rep)):
                                output[key] =\
                                    output[key].to(final_device)
                            if isinstance(output[key], dict):
                                for k in output[key]:
                                    output[key][k] =\
                                        output[key][k].to(final_device)
                    outputs.append(output)
            if 'output' in instruction:
                yield outputs

        self.set_params(saved_params)

        return
