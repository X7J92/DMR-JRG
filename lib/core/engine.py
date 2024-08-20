# from IPython import embed
#
# class Engine(object):
#     def __init__(self):
#         self.hooks = {}
#
#     def hook(self, name, state):
#
#         if name in self.hooks:
#             self.hooks[name](state)
#
#     def train(self, network, iterator, maxepoch, optimizer, scheduler):
#         state = {
#             'network': network,
#             'iterator': iterator,
#             'maxepoch': maxepoch,
#             'optimizer': optimizer,
#             'scheduler': scheduler,
#             'epoch': 0,
#             't': 0,
#             'train': True,
#         }
#
#
#
#         self.hook('on_start', state)
#         while state['epoch'] < state['maxepoch']:
#             self.hook('on_start_epoch', state)
#             for sample in state['iterator']:
#                 state['sample'] = sample
#                 self.hook('on_sample', state)
#
#                 def closure():
#                     loss,output,_,_,_,_ = state['network'](state['sample'])
#                     state['output'] = output
#                     state['loss'] = loss
#
#                     # print('Epoch{:d} {:d}/{:d}\nsum:{:.4f}\n overlap:{:.4f}\n order:{:.4f}\n'.format(
#                     #     state['epoch'],
#                     #     state['t'],
#                     #     len(state['iterator']),
#                     #     loss.item(),
#                     #     loss_overlap.item(),
#                     #     loss_order.item()
#                     # ))
#
#                     loss.backward()
#                     self.hook('on_forward', state)
#                     # to free memory in save_for_backward
#                     state['output'] = None
#                     state['loss'] = None
#                     return loss
#
#                 state['optimizer'].zero_grad()
#                 state['optimizer'].step(closure)
#                 self.hook('on_update', state)
#                 state['t'] += 1
#             state['epoch'] += 1
#             self.hook('on_end_epoch', state)
#         self.hook('on_end', state)
#         return state
#         # self.hook('on_start', state)
#         # while state['epoch'] < state['maxepoch']:
#         #     self.hook('on_start_epoch', state)
#         #     for sample in state['iterator']:
#         #         state['sample'] = sample
#         #         self.hook('on_sample', state)
#         #
#         #         def closure():
#         #             loss,output,_,_ = state['network'](state['sample'])
#         #             state['output'] = output
#         #             state['loss'] = loss
#         #
#         #             # print('Epoch{:d} {:d}/{:d}\nsum:{:.4f}\n overlap:{:.4f}\n order:{:.4f}\n'.format(
#         #             #     state['epoch'],
#         #             #     state['t'],
#         #             #     len(state['iterator']),
#         #             #     loss.item(),
#         #             #     loss_overlap.item(),
#         #             #     loss_order.item()
#         #             # ))
#         #
#         #             loss.backward()
#         #             self.hook('on_forward', state)
#         #             # to free memory in save_for_backward
#         #             state['output'] = None
#         #             state['loss'] = None
#         #             return loss
#         #
#         #         state['optimizer'].zero_grad()
#         #         state['optimizer'].step(closure)
#         #         self.hook('on_update', state)
#         #         state['t'] += 1
#         #     state['epoch'] += 1
#         #     self.hook('on_end_epoch', state)
#         # self.hook('on_end', state)
#         # return state
#     def test(self, network, iterator, split):
#         state = {
#             'network': network,
#             'iterator': iterator,
#             'split': split,
#             't': 0,
#             'train': False,
#         }
#
#         self.hook('on_test_start', state)
#         for sample in state['iterator']:
#             state['sample'] = sample
#             self.hook('on_test_sample', state)
#
#             def closure():
#                 loss, output,video,text,video_l,sentence_number = state['network'](state['sample'])
#                 state['video'] = video
#                 state['text'] = text
#                 state['sentence_number'] = sentence_number
#                 state['output'] = output
#                 state['loss'] = loss
#                 state['video_l'] = video_l
#                 self.hook('on_test_forward', state)
#                 # to free memory in save_for_backward
#                 state['output'] = None
#                 state['loss'] = None
#                 state['video'] = None
#                 state['text'] = None
#                 state['video_l'] = None
#                 state['sentence_number'] = None
#             closure()
#             state['t'] += 1
#         self.hook('on_test_end', state)
#         return state
#
#     # def test(self, network, iterator, split):
#     #     state = {
#     #         'network': network,
#     #         'iterator': iterator,
#     #         'split': split,
#     #         't': 0,
#     #         'train': False,
#     #     }
#     #
#     #     self.hook('on_test_start', state)
#     #     for sample in state['iterator']:
#     #         state['sample'] = sample
#     #         self.hook('on_test_sample', state)
#     #
#     #         def closure():
#     #             loss, output, sims_g, sentence_number = state['network'](state['sample'])
#     #             state['sims_g'] = sims_g
#     #             state['sentence_number'] = sentence_number
#     #             state['output'] = output
#     #             state['loss'] = loss
#     #
#     #             self.hook('on_test_forward', state)
#     #             # to free memory in save_for_backward
#     #             state['output'] = None
#     #             state['loss'] = None
#     #             state['sims_g'] = None
#     #             state['sentence_number'] = None
#     #
#     #         closure()
#     #         state['t'] += 1
#     #     self.hook('on_test_end', state)
#     #     return state

from IPython import embed

class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):

        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer, scheduler):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': 0,
            't': 0,
            'train': True,
        }



        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss,output,_,_,_,_ = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss

                    # print('Epoch{:d} {:d}/{:d}\nsum:{:.4f}\n overlap:{:.4f}\n order:{:.4f}\n'.format(
                    #     state['epoch'],
                    #     state['t'],
                    #     len(state['iterator']),
                    #     loss.item(),
                    #     loss_overlap.item(),
                    #     loss_order.item()
                    # ))

                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state
        # self.hook('on_start', state)
        # while state['epoch'] < state['maxepoch']:
        #     self.hook('on_start_epoch', state)
        #     for sample in state['iterator']:
        #         state['sample'] = sample
        #         self.hook('on_sample', state)
        #
        #         def closure():
        #             loss,output,_,_ = state['network'](state['sample'])
        #             state['output'] = output
        #             state['loss'] = loss
        #
        #             # print('Epoch{:d} {:d}/{:d}\nsum:{:.4f}\n overlap:{:.4f}\n order:{:.4f}\n'.format(
        #             #     state['epoch'],
        #             #     state['t'],
        #             #     len(state['iterator']),
        #             #     loss.item(),
        #             #     loss_overlap.item(),
        #             #     loss_order.item()
        #             # ))
        #
        #             loss.backward()
        #             self.hook('on_forward', state)
        #             # to free memory in save_for_backward
        #             state['output'] = None
        #             state['loss'] = None
        #             return loss
        #
        #         state['optimizer'].zero_grad()
        #         state['optimizer'].step(closure)
        #         self.hook('on_update', state)
        #         state['t'] += 1
        #     state['epoch'] += 1
        #     self.hook('on_end_epoch', state)
        # self.hook('on_end', state)
        # return state
    def test(self, network, iterator, split):
        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train': False,
        }

        self.hook('on_test_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_test_sample', state)

            def closure():
                loss, output,video,text,video_l,sentence_number,video_idxs = state['network'](state['sample'])
                state['video'] = video
                state['text'] = text
                state['sentence_number'] = sentence_number
                state['output'] = output
                state['loss'] = loss
                state['video_l'] = video_l
                state['video_idxs'] = video_idxs
                self.hook('on_test_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None
                state['video'] = None
                state['text'] = None
                state['video_l'] = None
                state['sentence_number'] = None
                state['video_idxs'] = None
            closure()
            state['t'] += 1
        self.hook('on_test_end', state)
        return state

    def test(self, network, iterator, split):
        state = {
            'network': network,
            'iterator': iterator,
            'split': split,
            't': 0,
            'train': False,
        }

        self.hook('on_test_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_test_sample', state)

            def closure():
                loss, output, sims_g, sentence_number = state['network'](state['sample'])
                state['sims_g'] = sims_g
                state['sentence_number'] = sentence_number
                state['output'] = output
                state['loss'] = loss

                self.hook('on_test_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None
                state['sims_g'] = None
                state['sentence_number'] = None

            closure()
            state['t'] += 1
        self.hook('on_test_end', state)
        return state