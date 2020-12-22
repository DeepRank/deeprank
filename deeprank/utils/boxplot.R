# INPUT: rawdata.tsv 
#
#   label	caseID	modelID	target	DR	HS
#   Test	1YVB	1YVB_ranair-it0_4286	0	0.56818	4.04629
#   Test	1PPE	1PPE_ranair-it0_2999	0	0.56486	50.17506

library(ggplot2)

args = commandArgs(trailingOnly=TRUE)
input_FL = args[1] #'it0.rawdata.tsv'
outFL = args[2] #'boxplot.png'


data = read.csv(input_FL, sep = '\t')

data$target = as.character(data$target)
data$label = factor(data$label, levels =  c('Train','Valid','Test'))

print(data[0:5,])

font_size = 20
text_style = element_text(size=font_size, family="Helvetica", face="bold")
colormap = c('0' = 'ivory3', '1' = 'steelblue')

ggplot(data, aes(x=target, y = DR, fill = target)) + 
  geom_boxplot(width=0.2,alpha=0.7) +
  facet_grid(.~label ) + 
  scale_fill_manual(values = colormap) +
  theme_bw() +
  theme(legend.position = 'right', 
        plot.title= text_style,
        text= text_style,
        axis.text.x= element_text(size=font_size),
        axis.text.y= element_text(size=font_size)) +
  scale_x_discrete(name = 'Target')


ggsave(outFL, dpi = 100)
print(paste(outFL, 'generated.'))
