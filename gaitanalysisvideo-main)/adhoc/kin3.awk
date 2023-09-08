# Read time series record, convert time to percentage of gait cycle assuming each file contain records for 1 gait cycle 
# output group by file name and transpose (ankle, knee, hip) columns to rows
# sample input:
## AR Gait_P_WooSG_Free walk_02-11-2022_15-44-59_noAR_kine_1.csv,2.068360655737705,-3.666665527774285,-0.036727591557290586,4.03348607299022
## AR Gait_P_WooSG_Free walk_02-11-2022_15-44-59_noAR_kine_1.csv,2.076700819672131,-4.341897602921108,-0.4770920770756245,3.876229594377968
# sample output:
## ,,gait cycle,0.00,0.01,0.02, ...
## AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR_kine_1.csv,L,ankle,-3.061449887479,-3.2031806250082724,-3.2316941033647595, ...

function output() {
  PROCINFO["sorted_in"] = "@ind_str_asc"
  for (f in files) {
    if (index(f, "kine_1.csv") > 0) side = "L"; else side = "R";
    PROCINFO["sorted_in"] = "@ind_num_asc"

    if (cnt > 0) {
      printf(",,gait cycle")
      for (g in n_ankles) {
        printf(",%s",g)
      }
      printf("\n")
      cnt = 0  # print gait cycle percent once
    }

    printf("%s,%s,ankle",f,side)
    for (g in n_ankles) {
      printf(",%s",n_ankles[g][f])
    }
    printf("\n")

    printf("%s,%s,knee",f,side)
    for (g in n_knees) {
      printf(",%s",n_knees[g][f])
    }
    printf("\n")

    printf("%s,%s,hip",f,side)
    for (g in n_hips) {
      printf(",%s",n_hips[g][f])
    }
    printf("\n")
  }
}
function reset() {
  delete ankles
  delete knees
  delete hips
  cnt = 0
}
function reindex() {
  for (i in ankles) {
    gait_cycle_percent = sprintf("%.2f",(i-1)/(cnt-1))
    n_ankles[gait_cycle_percent][filename] = ankles[i]
    n_knees[gait_cycle_percent][filename] = knees[i]
    n_hips[gait_cycle_percent][filename] = hips[i]
  }
}
# change in filename and not first record, reindex arrays
!($1 in files) && NR > 1 {
  reindex()
  reset()
}
{
 filename = $1
 cnt = cnt + 1
 ankles[cnt] = $3
 knees[cnt] = $4
 hips[cnt] = $5
 files[filename] = cnt
}
END {
  reindex()
  output()
}
