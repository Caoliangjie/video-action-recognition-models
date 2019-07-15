import random
import math


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        rate (int): Sample rate of the crop.
    """

    def __init__(self, size, rate):
        self.size = size
        self.rate = rate

    def __call__(self, frame_indices):
        frame_count = len(frame_indices)
        sample_duration = self.size
        sample_rate = self.rate
        total_sample_length = sample_duration * sample_rate
        sample_frame_indices = list()
         
        begin_index = 0
         
        for index in range(sample_duration):
            sample_frame_index = begin_index + index * sample_rate
            sample_frame_index = sample_frame_index % int(frame_count)
            sample_frame_indices.append(frame_indices[sample_frame_index])
            
        return sample_frame_indices


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        rate (int): Sample rate of the crop.
    """

    def __init__(self, size, rate):
        self.size = size
        self.rate = rate

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        frame_count = len(frame_indices)
        sample_duration = self.size
        sample_rate = self.rate
        total_sample_length = sample_duration * sample_rate
        sample_frame_indices = list()
         
        rand_end = max(0, frame_count - total_sample_length)
        begin_index = rand_end // 2
         
        for index in range(sample_duration):
            sample_frame_index = begin_index + index * sample_rate
            sample_frame_index = sample_frame_index % int(frame_count)
            sample_frame_indices.append(frame_indices[sample_frame_index])
            
        return sample_frame_indices

    
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        rate (int): Sample rate of the crop.
    """

    def __init__(self, size, rate):
        self.size = size
        self.rate = rate

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        frame_count = len(frame_indices)
        sample_duration = self.size
        sample_rate = self.rate
        total_sample_length = sample_duration * sample_rate
        sample_frame_indices = list()
         
        rand_end = max(0, frame_count - total_sample_length)
        begin_index = random.randint(0, rand_end)
         
        for index in range(sample_duration):
            sample_frame_index = begin_index + index * sample_rate
            sample_frame_index = sample_frame_index % int(frame_count)
            sample_frame_indices.append(frame_indices[sample_frame_index])
             
        return sample_frame_indices
    
class TestTemporalUniformCrop(object):
    """Temporally crop the given frame indices at multiple locations.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
        rate (int): Sample rate of the crop.
    """

    def __init__(self, size, rate):
        self.size = size
        self.rate = rate

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        # NL-I3D method
        clip_index = frame_indices[0]
        clip_count = frame_indices[1]
        frame_count = frame_indices[2]
        
        frame_indices = list(range(1, frame_count + 1))
        sample_duration = self.size
        sample_rate = self.rate
        total_sample_length = sample_duration * sample_rate
        sample_frame_indices = list()
        
        step = frame_count / clip_count

        begin_index = int(clip_index * step) % int(frame_count)
         
        for index in range(sample_duration):
            sample_frame_index = begin_index + index * sample_rate
            sample_frame_index = sample_frame_index % int(frame_count)
            sample_frame_indices.append(frame_indices[sample_frame_index])

        return sample_frame_indices
