
set graphics off

import excel "results/FigureA3_Robustness_check_pure_homophily_results.xlsx", sheet("Deepwalk") firstrow
egen ccode2 = group(order Model)
labmask ccode2, values(Model)
graph box Beta_coefficient, over(Embedding_Type) over(ccode2)  asyvar yline(0.0,lpattern("-") lcolor(black))  ylabel(-0.2(0.1)0.3,angle(horizontal)) graphregion(fcolor(white))
graph export "FigureA3(a).png", replace
clear

import excel "results/FigureA3_Robustness_check_pure_homophily_results.xlsx", sheet("Node2vec") firstrow
egen ccode2 = group(order Model)
labmask ccode2, values(Model)
graph box Beta_coefficient, over(Embedding_Type) over(ccode2)  asyvar yline(0.0,lpattern("-") lcolor(black)) ylabel(-0.2(0.1)0.3,angle(horizontal)) graphregion(fcolor(white))
graph export "FigureA3(b).png", replace
clear

import excel "results/FigureA4_Robustness_check_positive_peer_effect_results.xlsx", sheet("Deepwalk") firstrow
egen ccode2 = group(order Model)
labmask ccode2, values(Model)
graph box Beta_coefficient, over(Embedding_Type) over(ccode2)  asyvar yline(0.2,lpattern("-") lcolor(black))  ylabel(0.0(0.1)0.4,angle(horizontal)) graphregion(fcolor(white))
graph export "FigureA4(a).png", replace
clear

import excel "results/FigureA4_Robustness_check_positive_peer_effect_results.xlsx", sheet("Node2vec") firstrow
egen ccode2 = group(order Model)
labmask ccode2, values(Model)
graph box Beta_coefficient, over(Embedding_Type) over(ccode2)  asyvar yline(0.2,lpattern("-") lcolor(black))  ylabel(0.0(0.1)0.4,angle(horizontal)) graphregion(fcolor(white))
graph export "FigureA4(b).png", replace
clear

