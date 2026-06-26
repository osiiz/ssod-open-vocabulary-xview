import matplotlib.pyplot as plt
import re
import argparse
import os


def plot_loss_and_map(log_file, output_image, filter_start=1, filter_end=None):
    runs = []
    # Separamos explícitamente las iteraciones de train y de val
    current_run = {"epoch_data": [], "train_iter_data": [], "val_iter_data": []}

    with open(log_file, "r") as f:
        for line in f:
            if "[INFO] Argumentos:" in line:
                if (
                    current_run["epoch_data"]
                    or current_run["train_iter_data"]
                    or current_run["val_iter_data"]
                ):
                    runs.append(current_run)
                current_run = {
                    "epoch_data": [],
                    "train_iter_data": [],
                    "val_iter_data": [],
                }
                continue

            # Capturar iteraciones de entrenamiento
            match_iter_train = re.search(
                r"Epoch (\d+) \| iter (\d+)/(\d+) \| train_loss=([0-9.]+)", line
            )
            if match_iter_train:
                ep = int(match_iter_train.group(1))
                it = int(match_iter_train.group(2))
                tot = int(match_iter_train.group(3))
                loss = float(match_iter_train.group(4))
                current_run["train_iter_data"].append((ep, it, tot, loss))
                continue

            # Capturar iteraciones de validación
            match_iter_val = re.search(
                r"Epoch (\d+) \| iter (\d+)/(\d+) \| val_loss=([0-9.]+)", line
            )
            if match_iter_val:
                ep = int(match_iter_val.group(1))
                it = int(match_iter_val.group(2))
                tot = int(match_iter_val.group(3))
                loss = float(match_iter_val.group(4))
                current_run["val_iter_data"].append((ep, it, tot, loss))
                continue

            # Capturar el resumen de la época completa
            match_epoch = re.search(
                r"Epoch (\d+)/\d+ \| train_loss=([0-9.]+) \| val_loss=([0-9.]+) \| (?:AP50|mAP50|mAP)=([0-9.]+) \| lr=([0-9.eE+-]+)",
                line,
            )
            if match_epoch:
                ep = int(match_epoch.group(1))
                tl = float(match_epoch.group(2))
                vl = float(match_epoch.group(3))
                mp = float(match_epoch.group(4))
                lr = float(match_epoch.group(5))
                current_run["epoch_data"].append((ep, tl, vl, mp, lr))
                continue

    if (
        current_run["epoch_data"]
        or current_run["train_iter_data"]
        or current_run["val_iter_data"]
    ):
        runs.append(current_run)

    runs = [
        r for r in runs if r["epoch_data"] or r["train_iter_data"] or r["val_iter_data"]
    ]

    if not runs:
        print("Error: No se encontraron datos válidos en el log.")
        return

    if filter_start > 1 or filter_end is not None:
        for run in runs:
            if filter_end is not None:
                run["epoch_data"] = [
                    x for x in run["epoch_data"] if filter_start <= x[0] <= filter_end
                ]
                run["train_iter_data"] = [
                    x
                    for x in run["train_iter_data"]
                    if filter_start <= x[0] <= filter_end
                ]
                run["val_iter_data"] = [
                    x
                    for x in run["val_iter_data"]
                    if filter_start <= x[0] <= filter_end
                ]
            else:
                run["epoch_data"] = [
                    x for x in run["epoch_data"] if x[0] >= filter_start
                ]
                run["train_iter_data"] = [
                    x for x in run["train_iter_data"] if x[0] >= filter_start
                ]
                run["val_iter_data"] = [
                    x for x in run["val_iter_data"] if x[0] >= filter_start
                ]

        # Limpiar "runs" vacías que pudieran quedar tras el filtrado
        runs = [
            r
            for r in runs
            if r["epoch_data"] or r["train_iter_data"] or r["val_iter_data"]
        ]

        if not runs:
            print(
                f"Error: Después de filtrar (épocas {filter_start} a {filter_end if filter_end else 'final'}), no quedan datos para pintar."
            )
            return

    timelines = []
    current_tl = runs[0]
    timelines.append((current_tl, "Entrenamiento Original"))

    for i in range(1, len(runs)):
        run = runs[i]

        # Buscamos la época más baja de forma segura en los 3 conjuntos de datos
        start_epochs = []
        if run["epoch_data"]:
            start_epochs.append(run["epoch_data"][0][0])
        if run["train_iter_data"]:
            start_epochs.append(run["train_iter_data"][0][0])
        if run["val_iter_data"]:
            start_epochs.append(run["val_iter_data"][0][0])

        start_epoch = min(start_epochs)

        filtered_epoch_data = [
            x for x in current_tl["epoch_data"] if x[0] < start_epoch
        ]
        filtered_train_iter = [
            x for x in current_tl["train_iter_data"] if x[0] < start_epoch
        ]
        filtered_val_iter = [
            x for x in current_tl["val_iter_data"] if x[0] < start_epoch
        ]

        new_tl = {
            "epoch_data": filtered_epoch_data + run["epoch_data"],
            "train_iter_data": filtered_train_iter + run["train_iter_data"],
            "val_iter_data": filtered_val_iter + run["val_iter_data"],
        }

        timelines.append((new_tl, f"Reanudado desde época {start_epoch}"))
        current_tl = new_tl

    name, ext = os.path.splitext(output_image)
    output_files = []

    for i in range(len(timelines)):
        if i == len(timelines) - 1:
            output_files.append(output_image)
        else:
            output_files.append(f"{name}_part{i+1}{ext}")

    for idx, ((tl_data, title_suffix), out_file) in enumerate(
        zip(timelines, output_files)
    ):
        fig, ax1 = plt.subplots(figsize=(12, 7))

        color_train = "steelblue"
        color_val = "darkorange"
        color_iter = "lightskyblue"
        color_iter_val = "moccasin"

        ax1.set_xlabel("Época", fontsize=12)
        ax1.set_ylabel("Loss (Pérdida)", fontsize=12)

        lines_for_legend = []
        labels_for_legend = []

        # Pintar iteraciones de entrenamiento
        if tl_data["train_iter_data"]:
            iter_x = [
                ep - 1 + (it / tot) for ep, it, tot, _ in tl_data["train_iter_data"]
            ]
            iter_y = [loss for _, _, _, loss in tl_data["train_iter_data"]]

            (line_iter,) = ax1.plot(
                iter_x,
                iter_y,
                color=color_iter,
                alpha=0.4,
                linewidth=1.5,
                label="Train Loss (iter)",
            )
            lines_for_legend.append(line_iter)
            labels_for_legend.append(line_iter.get_label())

        # Pintar iteraciones de validación
        if tl_data["val_iter_data"]:
            iter_x_val = [
                ep - 1 + (it / tot) for ep, it, tot, _ in tl_data["val_iter_data"]
            ]
            iter_y_val = [loss for _, _, _, loss in tl_data["val_iter_data"]]

            (line_iter_val,) = ax1.plot(
                iter_x_val,
                iter_y_val,
                color=color_iter_val,
                alpha=0.4,
                linewidth=1.5,
                label="Val Loss (iter)",
            )
            lines_for_legend.append(line_iter_val)
            labels_for_legend.append(line_iter_val.get_label())

        lr_str = "Desconocido"

        if tl_data["epoch_data"]:
            epochs = [x[0] for x in tl_data["epoch_data"]]
            train_loss = [x[1] for x in tl_data["epoch_data"]]
            val_loss = [x[2] for x in tl_data["epoch_data"]]
            map_scores = [x[3] for x in tl_data["epoch_data"]]
            lrs = [x[4] for x in tl_data["epoch_data"]]

            lr_sequence = []
            for lr in lrs:
                if not lr_sequence or lr_sequence[-1] != lr:
                    lr_sequence.append(lr)

            lr_str = " ➔ ".join([f"{lr:g}" for lr in lr_sequence])

            # Las líneas sólidas resumen van encima de las iteraciones por el orden de pintado
            (line_train_epoch,) = ax1.plot(
                epochs,
                train_loss,
                label="Train Loss (época)",
                marker="o",
                markersize=6,
                color=color_train,
                linewidth=2,
            )
            (line_val_epoch,) = ax1.plot(
                epochs,
                val_loss,
                label="Validation Loss (época)",
                marker="s",
                markersize=6,
                color=color_val,
                linewidth=2,
            )

            lines_for_legend.extend([line_train_epoch, line_val_epoch])
            labels_for_legend.extend(
                [line_train_epoch.get_label(), line_val_epoch.get_label()]
            )

            ax1.tick_params(axis="y")
            ax1.grid(True, linestyle="--", alpha=0.7)

            ax2 = ax1.twinx()
            color_map = "mediumseagreen"
            ax2.set_ylabel(
                "AP50 (Average Precision)",
                color=color_map,
                fontsize=12,
                fontweight="bold",
            )

            map_epochs = [e for e, m in zip(epochs, map_scores) if m > 0]
            map_scores_nonzero = [m for m in map_scores if m > 0]

            if map_scores_nonzero:
                (line_map,) = ax2.plot(
                    map_epochs,
                    map_scores_nonzero,
                    label="AP50 (IoU=0.50)",
                    color=color_map,
                    linestyle="-",
                    marker="D",
                    linewidth=2.5,
                )
                ax2.tick_params(axis="y", labelcolor=color_map)

                best_map = max(map_scores_nonzero)
                best_epoch = map_epochs[map_scores_nonzero.index(best_map)]
                line_best = ax2.axvline(
                    x=best_epoch,
                    color="red",
                    linestyle="--",
                    alpha=0.6,
                    linewidth=2,
                    label=f"Best Epoch ({best_epoch})",
                )
                ax2.scatter(best_epoch, best_map, color="red", s=120, zorder=5)

                lines_for_legend.extend([line_best, line_map])
                labels_for_legend.extend([line_best.get_label(), line_map.get_label()])

        if lines_for_legend:
            # Ampliado a 4 o 5 columnas para que la leyenda entre bien
            ax1.legend(
                lines_for_legend,
                labels_for_legend,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=3,
            )

        plt.title(
            f"Evolución de Pérdidas vs AP50 ({title_suffix})\nHistorial LR: {lr_str}",
            fontsize=15,
            pad=15,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"Gráfico generado correctamente: {out_file} (LR: {lr_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training loss and AP50 from log file"
    )
    parser.add_argument("log_file", help="Path to training log file")
    parser.add_argument("output_image", help="Path to output image file")
    parser.add_argument(
        "--filter_start", type=int, default=1, help="First epoch to include in the plot"
    )
    parser.add_argument(
        "--filter_end", type=int, help="Last epoch to include in the plot"
    )

    args = parser.parse_args()
    plot_loss_and_map(
        args.log_file,
        args.output_image,
        filter_start=args.filter_start,
        filter_end=args.filter_end,
    )
