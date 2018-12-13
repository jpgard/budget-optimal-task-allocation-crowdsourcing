library(tidyverse)
library(directlabels)
library(scales)

# a global theme used for all plots
plot_theme <- theme_bw() +
  theme(legend.position=c(0.1,0.15), 
        plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5),
        legend.background = element_blank())

results_1 <- read.csv("results_1.csv")
## fig 1
results_1 %>%
  dplyr::filter(method=="kos") %>%
  ggplot(aes(x=l, y=error, shape=factor(m), color=factor(m))) +
  stat_summary(geom="line", fun.y = mean) +
  stat_summary(geom="point", fun.y = mean) +
  stat_summary(geom="errorbar", fun.data = mean_se, width=0.3) +
  ggtitle("Error Probability Relative to Problem Size m", subtitle="Replication of Figure 1") +
  xlab("\u2113, number of queries per task") +
  ylab("Probability of Error") +
  guides(shape = guide_legend("m"), color=guide_legend("m")) +
  coord_trans(y = "log10") +
  plot_theme
ggsave("fig1.png", device = "png", width = 10, height = 8)

results_2a = 
  read.csv("results_2a.csv") %>% 
  dplyr::mutate(method = plyr::revalue(method, 
                                 c("kos" = "Iterative algorithm", 
                                   "majority_vote" = "Majority voting",
                                   "spectral" = "Singular vector")))
oracle_logscale <- function(l, q ) {log10(0.5 * exp(-(q + q^2)*l))}
results_2a %>%
  ggplot(aes(x = l, y = error, color = method, shape=method)) +
  stat_summary(geom="line", fun.y = mean) +
  stat_summary(geom="point", fun.y = mean) +
  stat_summary(geom="errorbar", fun.data = mean_se, width=0.3) +
  xlim(0, 30) +
  ylim(0, 1) +
  stat_function(fun=oracle_logscale, args=list("q" = 0.3), aes(color="Oracle estimator"))+
  scale_y_log10() +
  ggtitle("Error Probability Relative to Number of Queries \u2113", subtitle="Replication of Figure 2a") +
  ylab("Probability of Error") +
  xlab("\u2113, number of queries per task") +
  guides(shape=FALSE) +
  scale_color_manual(values=c("#a6cee3",
                              "#1f78b4",
                              "#b2df8a",
                              "#33a02c")) +
  plot_theme
ggsave("fig2a.png", device = "png", width = 10, height = 8)

results_2b <- 
  read.csv("results_2b.csv") %>% 
  dplyr::mutate(method = plyr::revalue(method, 
                                       c("kos" = "Iterative algorithm", 
                                         "majority_vote" = "Majority voting",
                                         "spectral" = "Singular vector")))
results_2b %>%
  ggplot(aes(x = q, y = error+1e-10, color = method, shape = method)) + 
  stat_summary(geom="line", fun.y = mean) +
  stat_summary(geom="point", fun.y = mean) +
  stat_summary(geom="errorbar", fun.data = mean_se, width=0.01) +
  xlab("Collective quality of the crowd, q")  +
  ylab("Probability of Error") +
  stat_function(fun=oracle_logscale, args=list("l" = 25), aes(color="Oracle estimator"))+
  scale_y_log10(limits = c(1e-6,1)) +
  ggtitle("Error Probability Relative to Crowd Quality q", subtitle="Replication of Figure 2b") +
  guides(shape=FALSE) +
  scale_color_manual(values=c("#a6cee3",
                              "#1f78b4",
                              "#b2df8a",
                              "#33a02c")) +
  plot_theme
ggsave("fig2b.png", device = "png", width = 10, height = 8)

results_3 <- read.csv("results_3.csv") %>% 
  dplyr::mutate(method = plyr::revalue(method, 
                                       c("kos" = "Iterative algorithm", 
                                         "majority_vote" = "Majority voting",
                                         "spectral" = "Singular vector")))
results_3 %>%
  ggplot(aes(x = easiness_beta, y = error, color = method)) + 
  stat_summary(geom="line", fun.y = mean) +
  stat_summary(geom="point", fun.y = mean) +
  stat_summary(geom="errorbar", fun.data = mean_se, width=0.3) +
  geom_hline(yintercept=10^oracle(l=25, q=0.3), color="#b2df8a") +
  geom_vline(xintercept=0, col="grey", linetype="dotted") + 
  scale_y_log10() + 
  scale_x_reverse(limits=c(25,0)) +
  ggtitle("Error Probability Relative To Task Easiness Parameter Beta", subtitle="Alpha fixed at 50") +
  xlab("Beta Parameter") +
  ylab("Probability of Error") +
  annotate("label", label="Low Easiness", x = 22.5, y = 0.03) +
  annotate("label", label="High Easiness", x = 2, y = 0.03) +
  scale_color_manual("Legend",values=c("#a6cee3",
                              "#1f78b4",
                              "#33a02c")) +
  plot_theme
ggsave("fig3.png", device = "png", width = 10, height = 8)


