# INPUT: hitrate_melted.tsv
#
#    label	rank	Methods	hit_rate
#    Test	1	hitRate_DR	0.01
#    Test	2	hitRate_DR	0.02

library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

hitrate_FL = args[1] #'hitrate_melted.tsv'
outFL = args[2] #'hitrate.png'

df_tmp = read.csv(hitrate_FL, sep = '\t')
df_tmp$label = factor(df_tmp$label, levels =  c('Train','Valid','Test'))

print(df_tmp[0:5,])

font_size = 20
text_style = element_text(size=font_size, family="Helvetica", face="bold")


ggplot(df_tmp) + aes_string(x='rank', y='hit_rate', color='label', linetype='Methods') + 
  facet_wrap('label~.', scales = 'free') +
  geom_line(size=1) + 
  labs(x= 'Top N models', y= 'Hit Rate') + 
  theme_bw()+
  theme('legend.position' = 'right', 
        'plot.title'= text_style,
        'text'= text_style,
        'axis.text.x'= element_text(size=font_size-6, angle = -45, vjust = 1, hjust= 0),
        'axis.text.y'= element_text(size=font_size-6)) +
  labs('colour'= "Sets") #+
#  coord_cartesian(xlim = c(0, 100))


ggsave(outFL, dpi = 100)
print(paste(outFL, 'generated.'))
