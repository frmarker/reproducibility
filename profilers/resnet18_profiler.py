import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
'''
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

# produce an prof object that contains all the relevant information about the profiling.
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

'''

# what operation is taking most of the cpu?
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

# export the result to chrome tracing format
#prof.export_chrome_trace("trace.json")


from torch.profiler import profile, tensorboard_trace_handler
with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("./profilers/log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

