# ELINAC: Autoencoder Approach for Electronic Invoices Data Clustering

This repository contains the source code for ELINAC.

Article published in the open access journal Applied Sciences: https://www.mdpi.com/2076-3417/12/6/3008

SCHULTE, Johannes P. et al. ELINAC: Autoencoder Approach for Electronic Invoices Data Clustering. Applied Sciences, v. 12, n. 6, p. 3008, 2022.

### Abstract

The most common method used to document monetary transactions in Brazil is by issuing electronic invoices (NF-e). The audit of electronic invoices is essential, which can be improved by using data mining solutions such as clustering and anomaly detection. However, applying them is not a simple task because NF-e data contains millions of records with noisy fields and nonstandard documents, especially short text descriptions. In addition to these challenges, it is costly to extract information from short texts to identify traces of mismanagement, embezzlement, commercial fraud, or tax evasion. Analyzing such data can be more effective when divided into well-defined groups. However, efficient solutions for clustering data with characteristics similar to NF-es have not been proposed in the literature yet. This article developed ELINAC, a service for clustering short-text data in NF-es that uses an automatic encoder to cluster data. ELINAC aids in auditing transactions documented in NF-e, clustering similar data by short-text descriptions, and making anomaly detection in numeric fields easier. For this, ELINAC explores how to model the automatic encoder without increasing a lot of calculation costs to suppress a large number of short text data. In the worst case, the results show that ELINAC efficiently groups data while performing three times faster than solutions adopted in the literature.
