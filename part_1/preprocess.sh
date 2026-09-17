#!/bin/bash
#SBATCH --job-name=synscapes_preprocess
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=rtx2080
#SBATCH --output=preprocess_%j.out
#SBATCH --chdir=/scratch.hpc/matteo.preda
#SBATCH --gres=gpu:1

set -euo pipefail

# ==============================================================================
# CONFIGURAZIONE - modifica questi valori secondo la tua struttura di cartelle
# ==============================================================================
VENV_NAME="industry_venv"
VENV_DIR="/scratch.hpc/matteo.preda/industry/AI_in_industry/${VENV_NAME}"
REQUIREMENTS_FILE="/scratch.hpc/matteo.preda/industry/AI_in_industry/requirements.txt"          # path a requirements.txt
SCRIPT_DIR="/scratch.hpc/matteo.preda/industry/AI_in_industry/part_1"             # cartella con preprocess_synscapes.py + labels.py

SYNSCAPES_ROOT="/scratch.hpc/matteo.preda/industry/synscapes/Synscapes"        # dataset originale
OUTPUT_ROOT="/scratch.hpc/matteo.preda/industry/synscapes/synscapes_processed"    # output preprocessato

D_MAX=80.0
# rtx2080: CPU singola quad-core -> allineo i worker alle CPU richieste sopra
WORKERS=4

# ==============================================================================
# 1) Virtual environment "industry": crealo solo se non esiste già
# ==============================================================================
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] Virtual environment '${VENV_NAME}' già presente in ${VENV_DIR}, salto la creazione."
else
    echo "[INFO] Virtual environment '${VENV_NAME}' non trovato, lo creo in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ==============================================================================
# 2) Installazione dipendenze da requirements.txt
#    --no-cache-dir per non saturare la quota utente (vedi istruzioni cluster)
# ==============================================================================
echo "[INFO] Aggiorno pip e installo le dipendenze da ${REQUIREMENTS_FILE}..."
pip3 install --no-cache-dir --upgrade pip
pip3 install --no-cache-dir -r "${REQUIREMENTS_FILE}"

# Il plugin FreeImage di imageio (necessario per leggere gli EXR) non è su
# PyPI e va scaricato a parte, una tantum. La home è condivisa fra le
# macchine, quindi una volta scaricato qui resta disponibile ovunque.
echo "[INFO] Verifico/scarico il plugin FreeImage per imageio (lettura EXR)..."
python3 -c "import imageio; imageio.plugins.freeimage.download()" || true

# ==============================================================================
# 3) Esecuzione dello script di preprocessing
# ==============================================================================
mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Avvio preprocessing Synscapes..."
echo "[INFO] input  = ${SYNSCAPES_ROOT}"
echo "[INFO] output = ${OUTPUT_ROOT}"

python3 "${SCRIPT_DIR}/preprocess_synscapes.py" \
    --synscapes-root "${SYNSCAPES_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --labels-py "${SCRIPT_DIR}/labels.py" \
    --d-max "${D_MAX}" \
    --workers "${WORKERS}"

deactivate
echo "[INFO] Preprocessing completato."
