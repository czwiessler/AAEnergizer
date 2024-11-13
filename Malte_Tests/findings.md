findings

1. attribute "doneChargingTime" missing/null in 4088 rows
2. the time in the weather data is UTC, so the actual time on premise is the given time -8 hours
3. 5c99728ff9af8b5022123831 and 5e7954b0f9af8b090600ec84 corrupted: negative difference between doneChargingTime and disconnectTime 3596 and 3592 seconds respectively
