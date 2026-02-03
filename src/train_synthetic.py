import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm
from .models import ProbGenerator, Critic, DeterministicGenerator
from .data import RingGMMTrunc, GridGMMTrunc

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_risk(G, D, loader, device, z_dim):
    risks = []
    with torch.no_grad():
        for real_batch, _ in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)
            z = torch.randn(batch_size, z_dim).to(device)
            fake_batch = G(z, sample=True)
            
            fake_out = D(fake_batch)
            real_out = D(real_batch)
            risk = torch.mean(real_out) - torch.mean(fake_out)
            risks.append(risk.item())
    return np.mean(risks)


def evaluate_risk_mc(G, D, loader, device, z_dim, mc_samples: int = 1):
    risks = []
    mc_samples = max(1, int(mc_samples))
    with torch.no_grad():
        for real_batch, _ in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)
            real_mean = D(real_batch).mean()

            r = 0.0
            for _ in range(mc_samples):
                z = torch.randn(batch_size, z_dim, device=device)
                fake_batch = G(z, sample=True)
                fake_mean = D(fake_batch).mean()
                r += (real_mean - fake_mean).item()
            risks.append(r / mc_samples)
    return float(np.mean(risks))


def pac_bayes_certificate(empirical_risk: float, kl_div: float, delta: float, diameter: float, n: int, lam: float | None = None) -> tuple[float, float]:
    a = kl_div + math.log(1.0 / delta)
    c = (diameter**2) / (4.0 * n)
    if lam is None:
        lam = math.sqrt(a / c)
    bound = empirical_risk + (a / lam) + (lam * c)
    return float(bound), float(lam)


def save_learning_curve(curve_rows: list[dict], outdir: str, dataset_name: str, sigma_0: float):
    out_path = Path(outdir) / "curves" / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    stem = f"curve_sigma_{sigma_0}"
    csv_file = out_path / f"{stem}.csv"
    tsv_file = out_path / f"{stem}.tsv"
    json_file = out_path / f"{stem}.json"
    png_file = out_path / f"{stem}.png"

    if not curve_rows:
        return

    fieldnames = list(curve_rows[0].keys())

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)

    with tsv_file.open("w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in curve_rows:
            f.write("\t".join(str(r[k]) for k in fieldnames) + "\n")

    json_file.write_text(json.dumps({"dataset": dataset_name, "sigma_0": float(sigma_0), "rows": curve_rows}, indent=2), encoding="utf-8")

    epochs = [r["epoch"] for r in curve_rows]
    emp = [r["emp_risk_epoch"] for r in curve_rows]
    cert = [r["cert_epoch"] for r in curve_rows]
    kl = [r["kl"] for r in curve_rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(epochs, emp, label="Empirical Risk (epoch avg)", color="green")
    axes[0].plot(epochs, cert, label="Certificate (epoch)", color="gray")
    axes[0].set_ylabel("Risk / Bound")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, kl, label="KL(ρ||π)", color="purple")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.suptitle(f"{dataset_name} learning curve (sigma_0={sigma_0})")
    fig.tight_layout()
    fig.savefig(png_file)
    plt.close(fig)

    print(f"Saved learning curve to {csv_file} and {png_file}")


def train_and_evaluate(
    dataset_name,
    sigma_0,
    epochs,
    n_train=10000,
    pretrain_epochs: int | None = None,
    mc_samples_train: int = 1,
    mc_samples_eval: int = 1,
    save_curves: bool = False,
    outdir: str = ".",
):
    device = get_device()
    print(f"[{dataset_name}] Training with sigma_0 = {sigma_0} on {device}")

    z_dim = 2
    hidden_dim = 64
    batch_size = 256
    critic_steps = 5
    lr = 0.001
    delta = 0.05
    
    if dataset_name == "Ring":
        data_gen = RingGMMTrunc(n_samples=n_train, batch_size=batch_size)
        diameter = 3.2 * 2
    elif dataset_name == "Grid":
        data_gen = GridGMMTrunc(n_samples=n_train, batch_size=batch_size)
        diameter = 8.2 * np.sqrt(2)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader, test_loader = data_gen.get_loaders()
    n = len(train_loader.dataset)
    lam = n / 1024
    
    rho_init = np.log(np.exp(sigma_0) - 1)

    det_G = DeterministicGenerator(z_dim, hidden_dim).to(device)
    det_D = Critic(2, hidden_dim).to(device)
    opt_det_G = optim.Adam(det_G.parameters(), lr=lr)
    opt_det_D = optim.Adam(det_D.parameters(), lr=lr)

    if pretrain_epochs is None:
        pretrain_epochs = max(2, epochs // 5)

    if pretrain_epochs > 0:
        det_pbar = tqdm(range(pretrain_epochs), desc=f"Pretrain {dataset_name}", leave=False)
        for _ in det_pbar:
            for real_batch, _ in train_loader:
                real_batch = real_batch.to(device)
                bs = real_batch.size(0)

                for _ in range(critic_steps):
                    z = torch.randn(bs, z_dim).to(device)
                    fake = det_G(z).detach()
                    loss_D = -(det_D(real_batch).mean() - det_D(fake).mean())
                    opt_det_D.zero_grad()
                    loss_D.backward()
                    opt_det_D.step()

                z = torch.randn(bs, z_dim).to(device)
                fake = det_G(z)
                loss_G = det_D(real_batch).mean() - det_D(fake).mean()
                opt_det_G.zero_grad()
                loss_G.backward()
                opt_det_G.step()

    G = ProbGenerator(z_dim, hidden_dim, rho_init=rho_init, rho_prior=rho_init).to(device)
    G.init_from_deterministic(det_G)
    D = Critic(2, hidden_dim).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)
    
    final_bound = 0.0
    curve_rows: list[dict] = []
    
    pbar = tqdm(range(epochs), desc=f"Train {dataset_name} s={sigma_0}", leave=False)
    
    for epoch in pbar:
        epoch_risks = []
        for real_batch, _ in train_loader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.size(0)

            for _ in range(critic_steps):
                z = torch.randn(current_batch_size, z_dim).to(device)
                fake_batch = G(z, sample=True).detach()
                real_out = D(real_batch)
                fake_out = D(fake_batch)
                
                loss_D = -(torch.mean(real_out) - torch.mean(fake_out))
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            with torch.no_grad():
                real_mean = D(real_batch).mean()

            mc_k = max(1, int(mc_samples_train))
            risk_mc = 0.0
            for _ in range(mc_k):
                z = torch.randn(current_batch_size, z_dim, device=device)
                fake_batch = G(z, sample=True)
                fake_mean = D(fake_batch).mean()
                risk_mc = risk_mc + (real_mean - fake_mean)
            current_risk = risk_mc / mc_k
            
            kl_div = G.kl_divergence()
            loss_G = current_risk + (1/lam) * kl_div
            
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            
            epoch_risks.append(float(current_risk.detach().cpu().item()))
                
        with torch.no_grad():
            emp_risk_epoch = float(np.mean(epoch_risks)) if epoch_risks else 0.0
            kl_val = float(G.kl_divergence().detach().cpu().item())
            final_bound, lam_star = pac_bayes_certificate(
                empirical_risk=emp_risk_epoch,
                kl_div=kl_val,
                delta=delta,
                diameter=diameter,
                n=n,
                lam=None,
            )
            if save_curves:
                curve_rows.append(
                    {
                        "epoch": int(epoch + 1),
                        "emp_risk_epoch": float(emp_risk_epoch),
                        "kl": float(kl_val),
                        "cert_epoch": float(final_bound),
                        "lam_star": float(lam_star),
                    }
                )
        pbar.set_postfix({"Cert": f"{final_bound:.4f}", "lam*": f"{lam_star:.2f}"})

    if save_curves:
        save_learning_curve(curve_rows, outdir=outdir, dataset_name=dataset_name, sigma_0=float(sigma_0))

    empirical_risk = evaluate_risk_mc(G, D, train_loader, device, z_dim, mc_samples=mc_samples_eval)
    test_risk = evaluate_risk_mc(G, D, test_loader, device, z_dim, mc_samples=mc_samples_eval)
    
    plt.figure(figsize=(5, 5))
    real_data = data_gen.train_data.numpy()
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.1, c='green', label='Real')
    z = torch.randn(1000, z_dim).to(device)
    with torch.no_grad():
        fake_data = G(z, sample=True).cpu().numpy()
    plt.scatter(fake_data[:, 0], fake_data[:, 1], alpha=0.3, c='blue', label='Generated')
    plt.title(f"{dataset_name} (sigma_0={sigma_0})")
    plt.axis('equal')
    plt.legend()
    plt.savefig(f"sample_{dataset_name}_sigma_{sigma_0}.png")
    plt.close()

    with torch.no_grad():
        kl_val = float(G.kl_divergence().detach().cpu().item())
        cert, _ = pac_bayes_certificate(
            empirical_risk=float(empirical_risk),
            kl_div=kl_val,
            delta=delta,
            diameter=diameter,
            n=n,
            lam=None,
        )
    return empirical_risk, test_risk, cert

def run_experiment(
    dataset_name,
    epochs: int = 10,
    n_train: int = 10000,
    pretrain_epochs: int | None = None,
    mc_samples_train: int = 1,
    mc_samples_eval: int = 1,
    outdir: str = ".",
    save_curves: bool = False,
):
    sigma_vals = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    emp_risks = []
    test_risks = []
    certificates = []
    
    print(f"\n=== Starting Experiment for {dataset_name} ===")
    
    for s in sigma_vals:
        e_risk, t_risk, bound = train_and_evaluate(
            dataset_name,
            s,
            epochs=epochs,
            n_train=n_train,
            pretrain_epochs=pretrain_epochs,
            mc_samples_train=mc_samples_train,
            mc_samples_eval=mc_samples_eval,
            save_curves=save_curves,
            outdir=outdir,
        )
        emp_risks.append(e_risk)
        test_risks.append(t_risk)
        certificates.append(bound)
        print(f"Result {dataset_name} sigma={s}: Emp={e_risk:.4f}, Test={t_risk:.4f}, Bound={bound:.4f}")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "sigma_0": float(s),
            "empirical_risk": float(er),
            "test_risk": float(tr),
            "certificate_95": float(cert),
        }
        for s, er, tr, cert in zip(sigma_vals, emp_risks, test_risks, certificates)
    ]

    csv_file = out_path / f"results_{dataset_name}.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sigma_0", "empirical_risk", "test_risk", "certificate_95"],
        )
        writer.writeheader()
        writer.writerows(rows)

    tsv_file = out_path / f"results_{dataset_name}.tsv"
    with tsv_file.open("w", encoding="utf-8") as f:
        f.write("sigma_0\tempirical_risk\ttest_risk\tcertificate_95\n")
        for r in rows:
            f.write(
                f"{r['sigma_0']}\t{r['empirical_risk']}\t{r['test_risk']}\t{r['certificate_95']}\n"
            )

    json_file = out_path / f"results_{dataset_name}.json"
    payload = {
        "dataset": dataset_name,
        "config": {
            "epochs": int(epochs),
            "n_train": int(n_train),
            "pretrain_epochs": None if pretrain_epochs is None else int(pretrain_epochs),
            "mc_samples_train": int(mc_samples_train),
            "mc_samples_eval": int(mc_samples_eval),
        },
        "rows": rows,
    }
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved numeric results to {csv_file}, {tsv_file}, {json_file}")

    x = np.arange(len(sigma_vals))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, emp_risks, width, label='Empirical Risk', color='green', alpha=0.8)
    plt.bar(x, test_risks, width, label='Test Risk', color='blue', alpha=0.8)
    plt.bar(x + width, certificates, width, label='95% Certificate', color='gray', alpha=0.8)
    
    plt.xlabel('sigma_0')
    plt.ylabel('Risk / Bound')
    plt.title(f'{dataset_name} Dataset Results')
    plt.xticks(x, [str(s) for s in sigma_vals])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_file = f"figure_risks_{dataset_name}.png"
    plt.savefig(output_file)
    print(f"Saved bar chart to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["Ring", "Grid"], help="Dataset to run")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per sigma_0 (default: 10)")
    parser.add_argument("--n-train", type=int, default=10000, help="Total samples before truncation/split (default: 10000)")
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Deterministic pretrain epochs (default: max(2, epochs//5)); set 0 to disable",
    )
    parser.add_argument(
        "--mc-samples-train",
        type=int,
        default=10,
        help="MC samples for E_{g~rho}[W_F] during training (default: 10; paper uses 100)",
    )
    parser.add_argument(
        "--mc-samples-eval",
        type=int,
        default=100,
        help="MC samples for E_{g~rho}[W_F] when reporting risks/certificate (default: 100)",
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for figures/results (default: .)")
    parser.add_argument("--save-curves", action="store_true", help="Save per-sigma learning curves (CSV/TSV/JSON + PNG)")
    args = parser.parse_args()
    
    run_experiment(
        args.dataset,
        epochs=args.epochs,
        n_train=args.n_train,
        pretrain_epochs=args.pretrain_epochs,
        mc_samples_train=args.mc_samples_train,
        mc_samples_eval=args.mc_samples_eval,
        outdir=args.outdir,
        save_curves=args.save_curves,
    )