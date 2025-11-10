
cd "$(dirname "$0")"
cd ..
# 设置随机种子以便复现 
SEED=42

# 运行Seq2Seq训练脚本
# 我们在这里定义了作业中建议的超参数
python src/train.py \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --embedding_dim 256 \
    --num_heads 8 \
    --feed_forward_dim 512 \
    --num_layers 4 \
    --num_epochs 10 \
    --seed $SEED \
    --save_path "checkpoints/seq2seq_model.pth" \
    --plot_path "results/seq2seq_loss_curve.png"

echo "Seq2Seq 训练完成。"
echo "模型保存在 checkpoints/seq2seq_model.pth"
echo "损失曲线图保存在 results/seq2seq_loss_curve.png"