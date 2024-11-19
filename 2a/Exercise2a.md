To prepare the dataset of charging sessions for analysis, we followed these steps:

1. **Data Ingestion:** 
    We loaded the dataset into a dataframe for easier manipulation and access. 
    Ensuring all columns were correctly typed, especially date and time columns 
    (`connectionTime`, `disconnectTime`, and `doneChargingTime`), was critical to facilitate analysis.

2. **Initial Inspection and Summary Statistics:** 
    We conducted an initial overview, reviewing summary statistics and scanning for any clear irregularities or patterns in the data that might indicate issues. 
    This also involved checking column values to ensure they aligned with expected ranges (e.g., `kWhDelivered` should be non-negative).

3. **Handling Missing Data:** 
    For missing values, I used different strategies based on the column:
   - **Time-related fields** (`connectionTime`, `disconnectTime`): 
       If missing, these would create gaps in understanding session flow, so I examined records with missing values. 
       If feasible, I interpolated times based on other sessions with similar attributes. 
       For sessions with completely missing timestamps, I excluded them if they were minor in count or appeared erroneous.
       We found that some 
     - **`doneChargingTime`**
         It is possible that some sessions do not have a `doneChargingTime` because the vehicle was not fully charged.
         We found that in some cases, the `disconnectTime` was before the `doneChargingTime`, which is not logical.
          In such cases, we corrected the `doneChargingTime` to be equal to the `disconnectTime`.
   - **Numerical fields** (`kWhDelivered`): 
       For missing values here, I used median or mean values per site and session context, 
       which helped in maintaining accuracy without introducing bias.
   - **Categorical fields** (`siteID`, `stationID`, etc.): 
       There were no missing values in these columns, but if there were, 
       I would have used mode imputation or excluded the records if the missing values were not recoverable.

4. **Dealing with Erroneous Data:**
   - **Outliers and Incorrect Entries:** 
       I identified extreme values in `kWhDelivered` and session durations (like abnormally high or negative values). 
       For such cases, I cross-checked against historical data for realistic ranges and replaced them as needed.
   - **Session Logic Errors:** 
       I ensured that timestamps made logical sense; for example, `disconnectTime` should not be before `connectionTime`. 
       Erroneous sequences were either corrected where possible or excluded if there was no clear way to recover the intended values.

5. **Timezone Consistency:** 
    Since sessions occurred across different time zones, I standardized timestamps to UTC to ensure consistent temporal analysis.

6. **Final Data Check:** 
    After cleaning, I re-checked for any remaining inconsistencies or missing values and created summary reports to confirm that the dataset was ready for the analysis stages. 
