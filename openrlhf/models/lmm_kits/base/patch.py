from abc import ABC, abstractmethod

class BasePatch(ABC):
    def __init__(self):
        self.loaded = False
    @abstractmethod
    def _add_get_inputs_embeds():
        '''
        Add a `get_inputs_embeds(*args,**kwargs)` method to the model class, 
        which embeds image embeddings into the text embeddings and return the results.
        '''
        return NotImplementedError

    @abstractmethod
    def _add_get_position_ids():
        '''
        Add a `get_posiiton_ids(*args,**kwargs)` method to the model class, 
        which return the position_ids of the given inputs.
        '''
        return NotImplementedError

    @abstractmethod
    def _add_offset_split_position_ids():
        '''
        Add a `offset_split_position_ids(*args,**kwargs)` method to the model class, 
        which offset the split position_ids to true position_ids.
        '''
        return NotImplementedError
    
    def _register_to_autoclass():
        '''
        Register the model to the corresponding AutoModel class and AutoConfig class. Used for non-hf customized model.
        '''
        return NotImplementedError
    
    def apply_liger_kernel():
        '''
        Apply liger kernel to the model.
        '''
        return NotImplementedError

    @classmethod
    @abstractmethod
    def _load_all_patches(cls):
        '''
        Load all patches.
        '''
        return NotImplementedError

    def load_all_patches(self,use_liger_kernel=False):
        if not self.loaded:
            self._load_all_patches()
            self.loaded = True
            if use_liger_kernel:
                self.apply_liger_kernel()
