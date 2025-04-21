checkpoint_config = dict(interval=1,
                         max_keep_ckpts=3)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'),
                dict(
                    type='NeptuneLoggerHook',
                    init_kwargs=dict(
                    project='ClaimCompanion/CoreModels',
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYjY0YWE5MC1kM2JmLTQ2MTYtODUwYy1lNTRkOWFjM2U2NzQifQ==',
                    name='FlashInterImage-L'
                    )
                ),
                ]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
