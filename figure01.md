# figure01
Daniel Fürth <br><br>Table of Contents:

- [Emission plot Fig. 1c](#emission-plot-fig.-1c)

## Emission plot Fig. 1c

``` r
files<-dir('data/spectra/emission', full.names = TRUE)

spectra<-read.table(files[1], sep='\t')
spectra$sample <- 1

for(i in seq_along(files)[-1]){
  spectra.tmp <-read.table(files[i], sep='\t')
  spectra.tmp$sample <- i
  spectra <- rbind(spectra, spectra.tmp)
}
```

``` r
spectra$fluor <- NA

spectra$fluor[spectra$sample %in% seq(9,17)] <- 594
spectra$fluor[!(spectra$sample %in% seq(9,17))] <- 488

spectra$status <- NA

status<-c('fluor', 'quench', 'fluor',
          'quench', 'quench', 'unquench',
          'unquench', 'unquench', 'unquench',
          'fluor', 'fluor', 'fluor',
          'quench', 'quench', 'quench',
          'unquench', 'unquench', 'unquench')
k<-1
for(i in unique(spectra$sample)){
  
  
  spectra$status[spectra$sample==i]<-status[i]

}


library(dplyr)
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
spec<- spectra %>% 
  group_by(fluor, status, V3) %>% 
  summarise(emission=mean(V4))
```

    `summarise()` has grouped output by 'fluor', 'status'. You can override using
    the `.groups` argument.

``` r
spec$emission[spec$fluor == 488]<-spec$emission[spec$fluor == 488]/max(spec$emission[spec$fluor == 488 & spec$status == 'fluor'])
spec$emission[spec$fluor == 594]<-spec$emission[spec$fluor == 594]/max(spec$emission[spec$fluor == 594 & spec$status == 'fluor'])

spec$emission <- spec$emission - mean(spec$emission[1:50])


col2hex <- function(cname)
{
  colMat <- col2rgb(cname)
  rgb(
    red=colMat[1,]/255,
    green=colMat[2,]/255,
    blue=colMat[3,]/255
  )
}

spec<- spec[-which(spec$status == 'unquench'),]

quartz(width = 130.1729/20, height = 83.8626/20)
par(yaxs='i', xaxs='i', mar=c(4,4,1,1))
plot(0,0, type='n', xlim=c(450,750), ylim=c(0,1), ylab='Emission', xlab="Wavelength (nm)", las=1, axes=F)
axis(2, las=1, at=c(0,0.5,1))

color <- c('green3', 'green4', 'red', 'red4')
unique_samples <- unique( paste(spec$fluor, spec$status) )
names(spec)<-c("fluor" ,   "status" ,  "wavelength"     ,  "emission")

peak.max <- numeric()
# Loop through each unique sample and create a polygon plot
for (sample_name in unique_samples) {
  # Subset the data for the current sample
  sample_data <- spec[paste(spec$fluor, spec$status) == sample_name, ]
  
  # Create a polygon plot for the current sample
  polygon(c(sample_data$wavelength, 
            sample_data$wavelength[nrow(sample_data)], 
            sample_data$wavelength[1],
            sample_data$wavelength[1]), 
          
          c(sample_data$emission, 0, 0, sample_data$emission[1]), 
          col = paste0( col2hex( color[match(sample_name, unique_samples)] ), '70' ),
          border = color[match(sample_name, unique_samples)], lwd=2, xpd=F )
  
  # Add a legend for sample names
  legend("topright", legend = unique_samples, fill = color)
  
  peak.max <- c(peak.max, sample_data$wavelength[which.max(sample_data$emission)] )
}

axis(1, at=seq(300,800,by=50))
```

![](figure01_files/figure-commonmark/unnamed-chunk-2-1.png)

Save the plot.

``` r
quartz.save(file="pdf/figure01_c.pdf", type='pdf')
```

    quartz_off_screen 
                    2 

Then compute quench ratio:

``` r
quench.ratio <- spec %>% group_by(fluor, status) %>% summarise(max(emission))
```

    `summarise()` has grouped output by 'fluor'. You can override using the
    `.groups` argument.

``` r
AZdye488<-quench.ratio$`max(emission)`[1]/quench.ratio$`max(emission)`[2]

AZdye594<-quench.ratio$`max(emission)`[3]/quench.ratio$`max(emission)`[4]

round(AZdye488, 2)
```

    [1] 5.74

``` r
round(AZdye594, 2)
```

    [1] 2.88
