import os 
from importlib import import_module
import torch.nn as nn 
import torch 

loss_class_dict = {}

def regist_loss(loss):
    loss_name = loss.__name__.lower()
    assert not loss_name in loss_class_dict, 'there is already registered dataset: %s in trainer_dict.' % loss_name
    loss_class_dict[loss_name] = loss

    return loss

def get_loss_class(loss_name:str):
    loss_name = loss_name.lower()
    return loss_class_dict[loss_name]

# import all python files in trainer folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('core.loss.{}'.format(module[:-3]))
del module


class Loss(nn.Module):
    def __init__(self, loss_string, tmp_info=[]):
        super().__init__()
        loss_string     = loss_string.replace(' ', '').lower()

        # parse loss string
        self.loss_list = []
        for single_loss in loss_string.split('+'):
            weight, name = single_loss.split('*')
            ratio = True if 'r' in weight else False
            weight = float(weight.replace('r', ''))

            if name in loss_class_dict:
                self.loss_list.append({ 'name': name,
                                        'weight': float(weight),
                                        'func': loss_class_dict[name](),
                                        'ratio': ratio})
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))
            
        # parse temporal information string
        self.tmp_info_list = []
        for name in tmp_info:
            lname=name.lower()
            if lname in loss_class_dict:
                self.tmp_info_list.append({ 'name': lname,
                                            'func': loss_class_dict[name]()})
            else:
                raise RuntimeError('undefined loss term: {}'.format(lname))


    def forward(self, input_data, model_output, data, module, loss_name=None, change_name=None, ratio=1.0):
        '''
        forward all loss and return as dict format.
        Args
            input_data   : input of the network (also in the data)
            model_output : output of the network
            data         : entire batch of data
            module       : dictionary of modules (for another network forward)
            loss_name    : (optional) choose specific loss with name
            change_name  : (optional) replace name of chosen loss
            ratio        : (optional) percentage of learning procedure for increase weight during training
        Return
            losses       : dictionary of loss
        '''
        loss_arg = (input_data, model_output, data, module)

        # calculate only specific loss 'loss_name' and change its name to 'change_name'
        if loss_name is not None:
            for single_loss in self.loss_list:
                if loss_name == single_loss['name']:
                    loss = single_loss['weight'] * single_loss['func'](*loss_arg)
                    if single_loss['ratio']: loss *= ratio
                    if change_name is not None:
                        return {change_name: loss}
                    return {single_loss['name']: loss}
            raise RuntimeError('there is no such loss in training losses: {}'.format(loss_name))

        # normal case: calculate all training losses at one time
        losses = {}
        for single_loss in self.loss_list:
            losses[single_loss['name']] = single_loss['weight'] * single_loss['func'](*loss_arg)
            if single_loss['ratio']: losses[single_loss['name']] *= ratio 

        # calculate temporal information
        tmp_info = {}
        for single_tmp_info in self.tmp_info_list:
            # don't need gradient
            with torch.no_grad():
                tmp_info[single_tmp_info['name']] = single_tmp_info['func'](*loss_arg)

        return losses, tmp_info
