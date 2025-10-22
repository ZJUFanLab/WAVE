setwd("/path/to/LUAD_up_down_genes")

data <- read.delim("TCGA-LUAD.star_counts.tsv", row.names = 1)

sample_types <- substr(colnames(data), 14, 15)
table(sample_types)

normal_samples <- colnames(data)[sample_types == "11"]
if (length(normal_samples) > 0) {
  print(paste("found", length(normal_samples), "normal samples"))
} else {
  print("not found normal samples")
}


library(org.Hs.eg.db)
library(dplyr)       

ensembl_ids <- gsub("\\..*", "", rownames(data))

gene_symbols <- mapIds(org.Hs.eg.db,
                       keys = ensembl_ids,
                       keytype = "ENSEMBL",
                       column = "SYMBOL")

data$gene_symbol <- gene_symbols[ensembl_ids]



data_clean <- data[!is.na(data$gene_symbol), ]


data_merged <- data_clean %>%
  group_by(gene_symbol) %>%
  summarise(across(everything(), mean, na.rm = TRUE)) %>%
  as.data.frame()


rownames(data_merged) <- data_merged$gene_symbol
data_merged <- data_merged[, -1]

write.csv(data_merged,file="TCGA-LUAD.star_counts-v2.csv")


genes <- read.csv("landmark_genes.txt",header=FALSE)$V1
data_filtered <- data_merged[rownames(data_merged) %in% genes, ]

sample_ids <- colnames(data_filtered)
group <- ifelse(substr(sample_ids, 14, 15) == "11", "Control", "Cancer")
group <- factor(group)

table(group)

library(limma)
design <- model.matrix(~0 + group)
colnames(design) <- levels(group)

expr_matrix <- as.matrix(data_filtered)

contrast_matrix <- makeContrasts(Cancer_vs_Control = Cancer - Control, levels = design)

fit <- lmFit(expr_matrix, design)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2)

deg_results <- topTable(fit2, coef = "Cancer_vs_Control", number = Inf, adjust = "BH")


write.csv(deg_results, file="LUAD_DEG.csv")
library(ggVolcano)

deg_results$geneSymbol <- rownames(deg_results)


deg_results <- add_regulate(deg_results, log2FC_name = "logFC",
                    fdr_name = "adj.P.Val",log2FC = 1, fdr = 0.05)


p <- ggvolcano(deg_results, x = "log2FoldChange", y = "padj",
               label = "geneSymbol", label_number = 30, 
               log2FC_cut = 1, FDR_cut = 0.05, output = FALSE,
               fills = c("#3AAE72","gray","#7A547F"),
               colors = c("#3AAE72","gray","#7A547F"),)+
  theme_classic()+
  theme(aspect.ratio = 0.9,
        axis.text.x=element_text(size=8),
        axis.text.y=element_text(size=8),
        axis.title.x=element_text(size=8),
        axis.title.y=element_text(size=8),
        legend.position="top")+
  theme(legend.text = element_text(size = 8))+
  guides(
    fill = guide_legend(title = NULL),
    color = guide_legend(title = NULL)
  )
p


ggsave(p, file="D:/Work/WAVE_v3/13.drug_connectivity/lung_cancer_vocalno.pdf")


logfc_threshold <- 1
padj_threshold <- 0.05

up_genes <- deg_results[deg_results$logFC > logfc_threshold & deg_results$adj.P.Val < padj_threshold, ]
down_genes <- deg_results[deg_results$logFC < -logfc_threshold & deg_results$adj.P.Val < padj_threshold, ]

cat("Upregulated genes:", nrow(up_genes), "\n")
cat("Downregulated genes:", nrow(down_genes), "\n")

write.csv(up_genes,"up_genes.csv")
write.csv(down_genes,"down_genes.csv")