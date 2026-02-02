# HIV-atherosclerosis-URMC

This project analyzes single-cell RNA-seq data to study immune cell behavior in HIV-associated atherosclerosis. The work was completed during a Summer Scholars research experience at the University of Rochester Medical Center (URMC) and focuses on understanding how chronic HIV infection affects immune signaling and cell state changes linked to cardiovascular disease risk.

Using single-cell RNA-seq datasets from HIV-positive and control samples, this project examines transcriptional differences across immune cell populations. The analysis pipeline was primarily implemented in Python using tools such as Scanpy for preprocessing and visualization, CellRank and Palantir for trajectory and pseudotime analysis, and MAGIC for gene expression smoothing. These methods were used to explore how immune cells transition between functional states and how these transitions differ in disease conditions.

In addition to trajectory analysis, this project applied network-based modeling using scBONITA to infer pathway activity and signaling logic from single-cell data. This allowed for the identification of pathways and genes that may contribute to immune dysfunction and inflammation in HIV-associated atherosclerosis. Results are visualized using UMAPs, lineage probability plots, gene expression trends, and pathway-level summaries.

Overall, this repository provides a reproducible single-cell analysis workflow for studying immune signaling changes in HIV. The project highlights how combining single-cell transcriptomics with trajectory inference and pathway modeling can provide insight into complex disease mechanisms and immune dysregulation.
