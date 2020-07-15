import os
import numpy as np
import sys
import threading
import sqlite3
import pycuda.driver
from params import get_params


def get_devices():
    return [num for num in range(pycuda.driver.Device.count())]


if __name__ == "__main__":

    datasize = 5000
    timeperiods = 10
    batchsize = 10000

    gpus = [0, 1, 2, 3]
    threads = []

    conn = sqlite3.connect("research.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS BatchesData")
    cursor.execute("DROP TABLE IF EXISTS TimePeriods")
    cursor.execute("DROP TABLE IF EXISTS SimulationResults")
    cursor.execute("DROP TABLE IF EXISTS SimulationApproaches")

    cursor.execute(
        "CREATE TABLE BatchesData ( \
            batchID INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, \
            start INTEGER NOT NULL, \
            end INTEGER NOT NULL, \
            finished INTEGER DEFAULT 0 \
        )"
    )

    inp_total = get_params()
    paramssize = len(inp_total)

    print(f"Searching in {paramssize} param pairs")

    for i in range(0, paramssize - batchsize + 1, batchsize):
        cursor.execute(
            "INSERT INTO BatchesData(start, end) VALUES(?, ?)",
            (i, min(i + batchsize, paramssize)),
        )

    cursor.execute(
        "CREATE TABLE TimePeriods ( \
            timeperiodID INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, \
            start INTEGER NOT NULL, \
            end INTEGER NOT NULL \
        )"
    )

    for tp_num in range(timeperiods):
        cursor.execute(
            "INSERT INTO TimePeriods(start, end) VALUES (?, ?)",
            ((tp_num + 1) * datasize, tp_num * datasize),
        )

    cursor.execute(
        "CREATE TABLE SimulationApproaches ( \
            simulationID INTEGER PRIMARY KEY, \
            order_pairs_param INTEGER NOT NULL,\
            order_start_size_param INTEGER NOT NULL,\
            order_step_size_param INTEGER NOT NULL,\
            interval_param INTEGER NOT NULL,\
            min_spread_param INTEGER NOT NULL,\
            min_position_param INTEGER NOT NULL,\
            max_position_param INTEGER NOT NULL \
        )"
    )

    for i in range(inp_total.shape[0]):
        cursor.execute(
            "INSERT OR IGNORE INTO SimulationApproaches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                i,
                str(inp_total[i, 0]),
                str(inp_total[i, 1]),
                str(inp_total[i, 2]),
                str(inp_total[i, 3] / 1000000),
                str(inp_total[i, 4] / 1000000),
                str(inp_total[i, 5]),
                str(inp_total[i, 6]),
            ),
        )

    cursor.execute(
        "CREATE TABLE SimulationResults ( \
            resultID INTEGER PRIMARY KEY AUTOINCREMENT, \
            simulationID INTEGER NOT NULL,\
            timeperiodID INTEGER NOT NULL, \
            result INTEGER NOT NULL, \
            FOREIGN KEY(simulationID) REFERENCES SimulationApproaches(simulationID), \
            FOREIGN KEY(timeperiodID) REFERENCES TimePeriods(timeperiodID)\
        )"
    )

    cursor.executescript(
        """
DROP INDEX IF EXISTS index_result;
CREATE INDEX index_result
ON SimulationResults (result);

DROP VIEW IF EXISTS SimulationStatistics;

CREATE VIEW SimulationStatistics AS
SELECT d.simulationID, AVG(diff) as MAE, Average, Minimum, Maximum, d2.order_pairs_param, d2.order_start_size_param, d2.order_step_size_param, d2.interval_param, d2.min_spread_param, d2.min_position_param, d2.max_position_param
FROM (
	SELECT s1.simulationID, ABS(s1.result - s2.average) AS diff, s2.Average, s2.Minimum, s2.Maximum
	FROM SimulationResults s1
	INNER JOIN (
		SELECT 
			simulationID, 
			AVG(result) AS Average, 
			MIN(result) AS Minimum, 
			MAX(result) AS Maximum
		FROM SimulationResults
		GROUP BY simulationID
	) s2
	ON s2.simulationID = s1.simulationID
) d
INNER JOIN (SELECT * FROM SimulationApproaches) d2
ON d2.simulationID = d.simulationID
GROUP BY d.simulationID
ORDER BY d.Average DESC;"""
    )

    conn.commit()

    print("Starting calculation")

    for gpu in gpus:
        comm = (
            sys.executable
            + " "
            + os.path.join(os.getcwd(), "gym_trading", "envs", "trading_sim_cuda.py")
            + f' --gpu {gpu} --data "hist_data/bitmex_1m.csv" --datasize {datasize*timeperiods} --batchsize {batchsize}'
        )

        thread = threading.Thread(target=os.system, args=[comm])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
