library(readr)
library(dplyr)
library(ggplot2)
library(forcats)

input_dir <- "/opt/ml/processing/input/"
filename <- Sys.glob(paste(input_dir, "*.csv", sep=""))
df <- read_csv(filename)

plot_data <- df %>%
    group_by(state) %>%
    count()

write_csv(plot_data, "/opt/ml/processing/csv/plot_data.csv")

plot <- plot_data %>%
    ggplot()+
    geom_col(aes(fct_reorder(state, n),
                 n,
                 fill = n)) + 
    coord_flip()+
    labs(
        title = "Number of people by state",
        subtitle = "From US-500 dataset",
        x = "State",
        y = "Number of people"
    )+
    theme_bw()

ggsave("/opt/ml/processing/images/census_plot.png", width = 10, height = 8, dpi = 100)