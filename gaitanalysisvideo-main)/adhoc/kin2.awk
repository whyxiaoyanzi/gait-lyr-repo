# Read time series record, group by file name and transpose (ankle, knee, hip) columns to rows
# sample input with comma delimiter:
## AR Gait_P_WooSG_Free walk_02-11-2022_15-44-59_noAR_kine_1.csv,2.068360655737705,-3.666665527774285,-0.036727591557290586,4.03348607299022
## AR Gait_P_WooSG_Free walk_02-11-2022_15-44-59_noAR_kine_1.csv,2.076700819672131,-4.341897602921108,-0.4770920770756245,3.876229594377968
# sample output:
## AR Gait_P_WooSG_Free walk_02-11-2022_15-44-59_noAR_kine_1.csv,L,ankle,-3.666665527774285,-4.341897602921108,-4.945523190712024,-5.473259424604407

function transpose() {
  printf("%s,%s,ankle",filename,side)
  for (i=1; i<=cnt; i++) printf(",%s", ankles[i])
  printf("\n")
  printf("%s,%s,knee",filename,side)
  for (i=1; i<=cnt; i++) printf(",%s", knees[i])
  printf("\n")
  printf("%s,%s,hip",filename,side)
  for (i=1; i<=cnt; i++) printf(",%s", hips[i])
  printf("\n")
}
function reset() {
  delete ankles
  delete knees
  delete hips
  cnt = 0
}
filename != $1 && NR > 1 {
  if (index(filename, "kine_1.csv") > 0) side = "L"; else side = "R"
  transpose()
  reset()
}
{filename = $1
 cnt = cnt + 1
 ankles[cnt] = $3
 knees[cnt] = $4
 hips[cnt] = $5
}
END {
  if (index(filename, "kine_1.csv") > 0) side = "L"; else side = "R"
  transpose()
}
