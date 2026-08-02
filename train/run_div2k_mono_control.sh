#!/bin/bash
# Controle de monocromia: o colapso cross-domain do decodificador e causado pela
# ESTATISTICA distante do dominio, ou simplesmente por treinar em imagens
# monocromaticas?
#
# Nos seis dominios reais as duas coisas estao perfeitamente confundidas: os tres
# que colapsam sao exatamente os tres monocromaticos, e com n=6 nao sao separaveis.
#
# Desenho: dois bracos sobre AS MESMAS 800 imagens do DIV2K (imagens naturais, que
# a backbone nunca viu -- WACNN e STanH foram treinadas em OpenImages), diferindo
# em uma unica variavel:
#     div2k_color   RGB original
#     div2k_gray    o mesmo conteudo com R=G=B
# Tudo o mais e identico: mesma particao, mesmo agendamento, mesmos warm-starts.
#
#   gray colapsa e color nao   -> a causa e a monocromia
#   os dois se comportam igual -> a monocromia nao basta; a distancia volta ao centro
#
# Hiperparametros casados com train/run_spectrum.sh (batch 16, patch 256,
# --save_delta, mesmo mapa lambda->derivacao). A UNICA diferenca e o numero de
# epocas: 229 em vez de 20, para igualar os ~160k passes de imagem dos dominios,
# que tem 8000 imagens contra as 700 daqui. A diversidade de dados continua menor
# e isso deve ser declarado ao reportar.
#
# Ordem de execucao: alternada por lambda, os dois bracos lado a lado. Assim uma
# interrupcao no meio ainda deixa pares completos, que e a comparacao que importa.
#
# Uso:  nohup bash train/run_div2k_mono_control.sh > logs/div2k_mono_control.log 2>&1 &
set -u
export PYTHONPATH=src
export PATH="$HOME/miniconda3/envs/stanh/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=229
BATCH=16
PATCH=256
DERIV_DIR="models/original_paper/STanH/derivations"
LAMBDAS=("0.06305"  "0.02"      "0.25")
DERIVS=( "D11-A040" "D10-A040"  "D13-A040")
ARMS=("div2k_gray" "div2k_color")

mkdir -p logs
echo "[$(date '+%F %T')] CONTROLE DE MONOCROMIA — inicio"

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; warm="${DERIVS[$i]}"
  for arm in "${ARMS[@]}"; do
    save="models/${arm}_decoder"
    if [ -f "$save/lambda_${lam}_best.pth.tar" ]; then
      echo "[$(date '+%F %T')] SKIP $arm lambda=$lam"; continue
    fi
    mkdir -p "$save"
    echo "===== [$(date '+%F %T')] $arm  decoder  lambda=$lam  (warm $warm) ====="
    python -u train/train_xray_full.py --mode decoder --save_delta \
        --lmbda "$lam" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --dataset "datasets/${arm}" --patch_size "$PATCH" "$PATCH" \
        --save_dir "$save" --wandb_project "PIBIC_StanH_${arm}_decoder" \
        --init_stanh "$DERIV_DIR/${warm}.pth.tar"
  done
done

echo "[$(date '+%F %T')] TREINO CONCLUIDO — avaliando no Kodak (dominio cross)"
for arm in "${ARMS[@]}"; do
  python -u eval/eval_full.py --models_dir "models/${arm}_decoder" \
      --dataset datasets/kodak --limit 24 --entropy_estimation \
      --out_json "results/${arm}_decoder_on_cross_rd.json"
done

echo "[$(date '+%F %T')] CONCLUIDO"
echo "Leitura: delta casado por bpp contra results/kodak_rd.json nos dois bracos."
echo "O contraste gray-color e a medicao; o valor absoluto de cada um nao e"
echo "comparavel aos seis dominios por causa da diferenca de diversidade de dados."
