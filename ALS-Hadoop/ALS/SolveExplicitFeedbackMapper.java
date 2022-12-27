/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ALS;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.mortbay.log.Log;

import java.io.IOException;
import java.util.List;

/** Solving mapper that can be safely executed using multiple threads */
public class SolveExplicitFeedbackMapper extends SharingMapper<IntWritable,VectorWritable,IntWritable,VectorWritable,OpenIntObjectHashMap<Vector>> {
	private double lambda;
	private int numFeatures;
	private final VectorWritable uiOrmj = new VectorWritable();

	@Override
	OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) throws IOException {
		Configuration conf = ctx.getConfiguration();
		int numEntities = Integer.parseInt(conf.get("num_entities"));
		return readMatrixByRowsFromDistributedCache(numEntities, conf);
	}

	@Override
	protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
		lambda = Double.parseDouble(ctx.getConfiguration().get("lambda"));
		numFeatures = ctx.getConfiguration().getInt("num_features", -1);
	}

	@Override
	protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx) throws IOException, InterruptedException {
		OpenIntObjectHashMap<Vector> uOrM = getSharedInstance();
		uiOrmj.set(solveExplicit(ratingsWritable, uOrM, lambda, numFeatures));
		ctx.write(userOrItemID, uiOrmj);
	}
	
	protected OpenIntObjectHashMap<Vector> readMatrixByRowsFromDistributedCache(int numEntities, Configuration conf) throws IOException {
		IntWritable rowIndex = new IntWritable();
		VectorWritable row = new VectorWritable();

		OpenIntObjectHashMap<Vector> featureMatrix = numEntities > 0 ? new OpenIntObjectHashMap<Vector>(numEntities) : new OpenIntObjectHashMap<Vector>();

		Path[] cachedFiles = HadoopUtil.getCachedFiles(conf);
		LocalFileSystem localFs = FileSystem.getLocal(conf);

		for (Path cachedFile : cachedFiles) {
			SequenceFile.Reader reader = null;
			try {
				reader = new SequenceFile.Reader(localFs, cachedFile, conf);
				while (reader.next(rowIndex, row)) {
					featureMatrix.put(rowIndex.get(), row.get());
				}
			} finally {
				Closeables.close(reader, true);
			}
		}

		Preconditions.checkState(!featureMatrix.isEmpty(), "Feature matrix is empty");
		return featureMatrix;
	}
	
	protected Vector solveExplicit(VectorWritable ratingsWritable, OpenIntObjectHashMap<Vector> uOrM, double lambda, int numFeatures) {
		Vector ratings = ratingsWritable.get();

		List<Vector> featureVectors = Lists.newArrayListWithCapacity(ratings.getNumNondefaultElements());
		for (Vector.Element e : ratings.nonZeroes()) {
			int index = e.index();
			featureVectors.add(uOrM.get(index));
		}

		return AlternatingLeastSquaresSolver.solve(featureVectors, ratings, lambda, numFeatures);
	}
}
