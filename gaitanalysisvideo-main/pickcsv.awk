# optional input parameter:
# oLeft : override left pick (defafult = mid point)
# oRight : override right pick (default = mid point)
/LEFT/ {side="left"; resno=1; next}
/RIGHT/ {side="right"; totalLeft=resno-1; resno=1; next}
/^Cadence/ {results[side][resno] = $0 results[side][resno]; resno += 1; next}
/""/ {next}
/Average/ {next}
{ results[side][resno] = results[side][resno] "\n" $0 }
END {
  totalRight = resno-1
  if (oLeft > 0) pickLeft = oLeft; else pickLeft = int((totalLeft+1)/2);
  if (oRight > 0) pickRight = oRight; else pickRight = int((totalRight+1)/2);
  print "LEFT", pickLeft, totalLeft
  print results["left"][pickLeft]
  print "RIGHT", pickRight, totalRight
  print results["right"][pickRight]
}
