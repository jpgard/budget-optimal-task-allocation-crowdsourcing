# plots of beta distribution over range of params
library(tidyverse)
library(magrittr)
values = list()
alpha = 50

plot_theme <- theme_bw() +
  theme(legend.position=c(0.1,0.15), 
        plot.title = element_text(hjust = 0.5), 
        plot.subtitle = element_text(hjust = 0.5),
        legend.background = element_blank())

x = seq(0, 1, length.out = 1000)
for (beta in c(1e-6, 2.5, 5, 7.5, 10, 15, 20, 25)){
  values[[as.character(beta)]] <- dbeta(x, shape1=alpha, shape2=beta)
}
results <- dplyr::bind_cols(values)
results$x_val <- x
results %<>% tidyr::gather(key="beta", value="density", -x_val)

results %>% ggplot(aes(x = x_val, y = density, color=beta, group=beta)) +
  geom_line() +
  ggtitle(expression(Distribution~of~Task~Easiness~With~Varying~Easiness~Parameter~Beta~(alpha==50))) +
  geom_text(data=results%>%dplyr::group_by( beta)%>%filter(density==max(density)),
            aes(x=x_val, y=density+0.5, label=beta), show.legend = FALSE) +
  labs(color = expression(beta))+
  ylab("Density") +
  xlab("Task Easiness") +
  plot_theme
ggsave("easiness.png", device = "png", width=10, height=8)
