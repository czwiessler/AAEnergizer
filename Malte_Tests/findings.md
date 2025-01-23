findings

1. attribute "doneChargingTime" missing/null in 4088 rows
2. the time in the weather data is UTC, so the actual time on premise is the given time -8 hours
3. 5c99728ff9af8b5022123831 and 5e7954b0f9af8b090600ec84 corrupted: negative difference between doneChargingTime and disconnectTime 3596 and 3592 seconds respectively
4. there are 27 rows with negative difference between doneChargingTime and connectionTime, meaning that the EV is fully charged before it starts charging. These rows are altered in a way so that doneChargingTime = connectionTime
5. charging power concentrates to approx. 3 and 6 kw, with most of the data points revolving around these two values and some outliers ranging up way highere
6. spaceID and stationID are in a bijective relationship, meaning that each spaceID is assigned to exactly one stationID and vice versa