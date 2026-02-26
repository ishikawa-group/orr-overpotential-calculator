#!/usr/bin/env python3
from pathlib import Path

from nanoparticle.orr_overpotential_calculator import (
    calc_nanoparticle_orr_overpotential_by_site,
)


def main() -> None:
    input_structure = Path(
        "/Users/wakamiya/Documents/20260205/"
        "nnp-nanoparticle-activity-stability-calculation/"
        "nanoparticle_size_dependence/result/Pt55/structure/"
        "Pt_Cuboctahedron_55_clean.extxyz"
    )
    outdir = Path("/Users/wakamiya/Documents/20260205/temp/Pt55_by_site")
    outdir.mkdir(parents=True, exist_ok=True)

    summary = calc_nanoparticle_orr_overpotential_by_site(
        clean_nanoparticle_structure=str(input_structure),
        n_samples=1,
        outdir=str(outdir),
        calculator="esen-oc25",
        optimizer="LBFGSLineSearch",
        max_opt_steps=80,
        retry_optimizer="FIRE",
        random_seed=0,
        vacuum_size=8.0,
    )

    print("site types:", summary["site_detection"]["n_site_types"])
    print("expected jobs:", summary["expected_adsorption_calculations"])
    print("summary:", outdir / "summary_by_site.json")


if __name__ == "__main__":
    main()

