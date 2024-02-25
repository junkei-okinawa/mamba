source ~/miniforge3/etc/profile.d/conda.sh
conda activate mamba_cpu
python "mamba_generate_long_tensor.py" \
--question "Quante torri ha bologna? " \
--context "La torre degli Asinelli è una delle cosiddette due torri di Bologna, simbolo della città, situate in piazza di porta Ravegnana, all'incrocio tra le antiche strade San Donato (ora via Zamboni), San Vitale, Maggiore e Castiglione. Eretta, secondo la tradizione, fra il 1109 e il 1119 dal nobile Gherardo Asinelli, la torre è alta 97,20 metri, pende verso ovest per 2,23 metri e presenta all'interno una scalinata composta da 498 gradini. Ancora non si può dire con certezza quando e da chi fu costruita la torre degli Asinelli. Si presume che la torre debba il proprio nome a Gherardo Asinelli, il nobile cavaliere di fazione ghibellina al quale se ne attribuisce la costruzione, iniziata secondo una consolidata tradizione l'11 ottobre 1109 e terminata dieci anni dopo, nel 1119." \
--max-length 2048 \
--tokenizer "DeepMount00/Mamba-QA-ITA-790m" \
--model-name "DeepMount00/Mamba-QA-ITA-790m" \
--device "mps"
# --device "cpu"