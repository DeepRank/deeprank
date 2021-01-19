library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

input_FL = args[1] #'successrate_melted.tsv'
outFL = args[2] #'successrate.png'

df = read.csv(input_FL, sep = '\t')
df$label = factor(df$label, levels =  c('Train','Valid','Test'))

print(df[0:5,])

font_size = 20
text_style = element_text(size=font_size, family="Helvetica", face="bold")

ggplot(df) + 
  aes_string(x='rank', y='success_rate', color='label', linetype='Methods') + 
  facet_wrap('label ~.',  scales = 'free') + 
  geom_line(size=1) + 
  labs('x'= 'Top N models', 'y'= 'Success Rate') + 
  theme_bw() + 
  theme('legend.position'= 'right',
  'plot.title'= text_style,
  'text'= text_style,
  'axis.text.x'= element_text(size=font_size-6, angle = -45, vjust = 1, hjust= 0),
  'axis.text.y'= element_text(size=font_size-6)) +
  labs('colour'= "Sets") #change legend title to 'Sets'
#        scale_x_continuous(**{'breaks': breaks, 'labels': xlabels})

ggsave(outFL, dpi = 100)
print(paste(outFL, 'generated.'))
