set -euo pipefail

BATCH_SIZE=10
TOTAL_IMAGES=10
ROOT_DIR=results/

Y_SIGMA=0.05
DYS_MAX_ITER=500
ANNEALING_STEPS=500
data=ffhq
mkdir -p $ROOT_DIR/$data

tasks=( down_sampling
    gaussian_blur
    hdr
    inpainting_rand
    inpainting
    motion_blur
)

tasks=( phase_retrieval
)

for task in "${tasks[@]}"; do
  #for denoise_type in 'tweedie' 'ode'; do
  for denoise_type in 'tweedie'; do
    echo ====================================================================================
    echo Running PDHG for $task in $data with $denoise_type denoiser with sigma=$Y_SIGMA
    echo ====================================================================================
    python recover_inverse2.py --config-name default_$data.yaml \
      sampler=edm_pdhg \
      inverse_task=$task \
      save_dir=$ROOT_DIR/$data/$task/pdhg-$denoise_type \
      batch_size=$BATCH_SIZE \
      total_images=$TOTAL_IMAGES \
      inverse_task.operator.sigma=$Y_SIGMA \
      sampler.annealing_scheduler_config.num_steps=$ANNEALING_STEPS \
      inverse_task.admm_config.denoise.final_step=$denoise_type \
      inverse_task.admm_config.max_iter=$DYS_MAX_ITER \
      inverse_task.admm_config.dys.gamma=0.0075\
      inverse_task.admm_config.dys.lambda_schedule.activate=true \
      inverse_task.admm_config.dys.lambda_schedule.start=1 \
      inverse_task.admm_config.dys.lambda_schedule.end=1 \
      inverse_task.admm_config.dys.lambda_schedule.warmup=0 \
      inverse_task.admm_config.denoise.lgvd.num_steps=0
  done
done
