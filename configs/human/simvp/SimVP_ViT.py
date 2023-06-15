method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'vit'
hid_S = 64
hid_T = 256
N_T = 8
N_S = 4
# training
lr = 1e-4
batch_size = 4  # 4 x bs4 = bs16
drop_path = 0.1
sched = 'onecycle'